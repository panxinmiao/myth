//! Asynchronous ring-buffer readback stream.
//!
//! [`ReadbackStream`] provides a non-blocking, high-throughput pipeline for
//! reading GPU render-target data back to the CPU. It is designed for
//! continuous workloads such as video recording and AI training-data
//! generation, where per-frame blocking would bottleneck the GPU pipeline.
//!
//! # Architecture
//!
//! A fixed-size ring of staging buffers rotates between three roles:
//!
//! 1. **Write slot** — receives a GPU `copy_texture_to_buffer` command.
//! 2. **In-flight** — the GPU is executing the copy; a `map_async` callback
//!    will fire when the data is ready.
//! 3. **Ready** — the callback has fired; the CPU can read the mapped buffer.
//!
//! A bounded `flume` channel whose capacity equals the ring size provides
//! natural back-pressure: if the CPU falls behind, `try_submit()` returns
//! `Err(ReadbackError::RingFull)` instead of silently allocating memory.
//!
//! # Memory reuse
//!
//! A `free_pool` of `Vec<u8>` buffers is maintained internally. When a frame
//! is consumed via [`try_recv_into`](ReadbackStream::try_recv_into), the
//! caller's previous buffer is swapped in and recycled, achieving **steady-state
//! zero allocation** after the first few frames.
//!
//! # Dual submit API
//!
//! | Method | Blocking? | Use case |
//! |--------|-----------|----------|
//! | [`try_submit`](ReadbackStream::try_submit) | Never | Real-time streaming (frame drops OK) |
//! | [`submit_blocking`](ReadbackStream::submit_blocking) | When ring full | Offline recording (zero frame loss) |
//!
//! # Usage
//!
//! ```rust,ignore
//! let mut stream = ReadbackStream::new(&device, width, height, format, 3)?;
//! let mut buf = Vec::new();
//!
//! for _ in 0..100 {
//!     engine.update(dt);
//!     engine.render_active_scene();
//!     stream.try_submit(&device, &queue, &headless_texture)?;
//!
//!     if let Some(idx) = stream.try_recv_into(&mut buf)? {
//!         process(idx, &buf);
//!     }
//! }
//!
//! // Drain remaining in-flight frames.
//! let remaining = stream.flush(&device)?;
//! for frame in remaining {
//!     process(frame.frame_index, &frame.pixels);
//! }
//! ```

use std::collections::VecDeque;

use myth_core::RenderError;

// ============================================================================
// Error types
// ============================================================================

/// Error returned by [`ReadbackStream`] operations.
#[derive(Debug)]
pub enum ReadbackError {
    /// All ring-buffer slots are in-flight or pending on the CPU side.
    /// The caller should drain frames with [`ReadbackStream::try_recv`] or
    /// [`ReadbackStream::flush`] before submitting more.
    RingFull(usize),

    /// The GPU buffer mapping failed.
    MapFailed(String),

    /// The texture format does not support readback.
    UnsupportedFormat(wgpu::TextureFormat),

    /// The internal stashed-frame buffer exceeded its safety limit.
    StashFull(usize),
}

impl std::fmt::Display for ReadbackError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RingFull(n) => write!(
                f,
                "readback ring buffer is full — all {n} slots are in-flight"
            ),
            Self::MapFailed(msg) => write!(f, "buffer mapping failed: {msg}"),
            Self::UnsupportedFormat(fmt) => write!(f, "unsupported readback format: {fmt:?}"),
            Self::StashFull(limit) => write!(
                f,
                "stash exceeded limit of {limit} frames — consume with try_recv/try_recv_into"
            ),
        }
    }
}

impl std::error::Error for ReadbackError {}

impl From<ReadbackError> for myth_core::Error {
    fn from(e: ReadbackError) -> Self {
        myth_core::Error::Render(RenderError::ReadbackFailed(e.to_string()))
    }
}

// ============================================================================
// Frame data
// ============================================================================

/// A frame of pixel data returned by the readback stream.
pub struct ReadbackFrame {
    /// Tightly-packed pixel data (format matches the stream's texture format).
    pub pixels: Vec<u8>,
    /// Zero-based index of this frame in submission order.
    pub frame_index: u64,
}

// ============================================================================
// ReadbackStream
// ============================================================================

/// High-throughput, non-blocking readback pipeline based on a ring buffer.
///
/// See [module documentation](self) for architecture details and usage.
pub struct ReadbackStream {
    buffers: Vec<wgpu::Buffer>,
    /// Next ring-buffer slot to write into.
    write_idx: usize,
    /// Monotonically increasing frame counter (incremented on each submit).
    next_frame_index: u64,
    /// Number of staging buffers currently owned by the GPU / channel.
    in_flight_frames: usize,

    // Bounded channel: sender lives in `map_async` callbacks, receiver is
    // polled by the CPU via `try_recv()` / `flush()`.
    sender: flume::Sender<ReadySlot>,
    receiver: flume::Receiver<ReadySlot>,

    /// Frames that were extracted during back-pressure handling
    /// (`submit_blocking`) and have not yet been consumed by the caller.
    stashed_frames: VecDeque<ReadbackFrame>,
    max_stash_size: usize,

    /// Object pool of previously-used pixel buffers whose heap capacity is
    /// still live. Recycling these avoids per-frame allocation once the
    /// system reaches steady state.
    free_pool: Vec<Vec<u8>>,

    // Layout constants (computed once at construction).
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
    bytes_per_pixel: u32,
    unpadded_bytes_per_row: u32,
    padded_bytes_per_row: u32,
}

/// Payload sent through the channel when a buffer becomes mappable.
struct ReadySlot {
    /// Index into `self.buffers`.
    slot: usize,
    /// Frame index assigned at submission time.
    frame_index: u64,
    /// Result from `map_async` callback.
    result: Result<(), wgpu::BufferAsyncError>,
}

impl ReadbackStream {
    /// Creates a new readback stream with `buffer_count` ring-buffer slots.
    ///
    /// # Arguments
    ///
    /// * `device` — GPU device for buffer allocation.
    /// * `width` / `height` — Dimensions of the source render target.
    /// * `format` — Pixel format (must support `block_copy_size`).
    /// * `buffer_count` — Number of staging buffers (ring size). Typical
    ///   values are 2–4; higher counts tolerate more GPU-to-CPU latency at
    ///   the cost of VRAM.
    ///
    /// # Errors
    ///
    /// Returns [`ReadbackError::UnsupportedFormat`] if the format does not
    /// have a known block copy size.
    pub fn new(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
        buffer_count: usize,
        max_stash_size: usize,
    ) -> Result<Self, ReadbackError> {
        let bytes_per_pixel = format
            .block_copy_size(None)
            .ok_or(ReadbackError::UnsupportedFormat(format))?;

        let unpadded_bytes_per_row = width * bytes_per_pixel;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_bytes_per_row = unpadded_bytes_per_row.div_ceil(align) * align;
        let buffer_size = u64::from(padded_bytes_per_row) * u64::from(height);

        let buffers: Vec<wgpu::Buffer> = (0..buffer_count)
            .map(|i| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("ReadbackStream Slot {i}")),
                    size: buffer_size,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                })
            })
            .collect();

        let (sender, receiver) = flume::bounded(buffer_count);

        Ok(Self {
            buffers,
            write_idx: 0,
            next_frame_index: 0,
            in_flight_frames: 0,
            sender,
            receiver,
            stashed_frames: VecDeque::with_capacity(max_stash_size),
            max_stash_size,
            free_pool: Vec::with_capacity(buffer_count),
            width,
            height,
            format,
            bytes_per_pixel,
            unpadded_bytes_per_row,
            padded_bytes_per_row,
        })
    }

    // ====================================================================
    // Submit API
    // ====================================================================

    /// Submits a non-blocking copy from `texture` to the next ring-buffer slot.
    ///
    /// The copy command is recorded and submitted immediately. When the GPU
    /// completes the copy, a `map_async` callback pushes the slot index into
    /// the internal channel, making it available via [`try_recv`](Self::try_recv).
    ///
    /// # Back-pressure
    ///
    /// If all slots are in-flight, this method returns
    /// [`ReadbackError::RingFull`] **without blocking**. The caller may
    /// choose to skip the frame (real-time path) or drain via
    /// [`try_recv`](Self::try_recv).
    pub fn try_submit(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        texture: &wgpu::Texture,
    ) -> Result<(), ReadbackError> {
        if self.in_flight_frames >= self.buffers.len() {
            return Err(ReadbackError::RingFull(self.buffers.len()));
        }
        self.record_and_submit(device, queue, texture);
        Ok(())
    }

    /// Submits a copy, blocking when the ring buffer is full.
    ///
    /// When all ring-buffer slots are occupied, this method:
    /// 1. Blocks until the GPU completes at least one pending readback.
    /// 2. Extracts the completed frame into `stashed_frames` (using a
    ///    recycled buffer from `free_pool` when available).
    /// 3. Proceeds with the submission on the freed slot.
    ///
    /// This guarantees **zero frame loss** and is intended for offline
    /// workloads such as video export and dataset generation.
    ///
    /// # Safety on WASM
    ///
    /// `wgpu::PollType::wait_indefinitely()` will panic on WebAssembly.
    /// Use [`try_submit`](Self::try_submit) for WASM targets.
    ///
    /// # Stash limit
    ///
    /// `max_stash_size` caps the number of completed-but-unconsumed frames
    /// held in memory. If the stash is full, returns
    /// [`ReadbackError::StashFull`] to prevent unbounded memory growth.
    pub fn submit_blocking(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        texture: &wgpu::Texture,
    ) -> Result<(), ReadbackError> {
        if self.in_flight_frames >= self.buffers.len() {
            if self.stashed_frames.len() >= self.max_stash_size {
                return Err(ReadbackError::StashFull(self.max_stash_size));
            }

            device
                .poll(wgpu::PollType::wait_indefinitely())
                .map_err(|e| ReadbackError::MapFailed(format!("device.poll failed: {e}")))?;

            if let Ok(ready) = self.receiver.try_recv() {
                ready
                    .result
                    .map_err(|e| ReadbackError::MapFailed(e.to_string()))?;

                let mut pixels = self.alloc_pixel_buf();
                self.extract_pixels_into(&self.buffers[ready.slot], &mut pixels);
                self.stashed_frames.push_back(ReadbackFrame {
                    pixels,
                    frame_index: ready.frame_index,
                });
                self.in_flight_frames -= 1;
            }
        }

        self.try_submit(device, queue, texture)
    }

    // ====================================================================
    // Receive API
    // ====================================================================

    /// Returns the next ready frame without blocking, or `None` if no frame
    /// is available yet.
    ///
    /// This is the **allocating** receive path — a new `Vec<u8>` (or one
    /// from the internal pool) is returned for each frame. For zero-copy
    /// steady-state operation, prefer [`try_recv_into`](Self::try_recv_into).
    pub fn try_recv(&mut self) -> Result<Option<ReadbackFrame>, ReadbackError> {
        if let Some(frame) = self.stashed_frames.pop_front() {
            return Ok(Some(frame));
        }

        match self.receiver.try_recv() {
            Ok(ready) => {
                ready
                    .result
                    .map_err(|e| ReadbackError::MapFailed(e.to_string()))?;

                let mut pixels = self.alloc_pixel_buf();
                self.extract_pixels_into(&self.buffers[ready.slot], &mut pixels);
                self.in_flight_frames -= 1;
                Ok(Some(ReadbackFrame {
                    pixels,
                    frame_index: ready.frame_index,
                }))
            }
            Err(flume::TryRecvError::Empty | flume::TryRecvError::Disconnected) => Ok(None),
        }
    }

    /// Zero-allocation receive: writes pixel data into a caller-supplied buffer.
    ///
    /// On success the previous contents of `output` are swapped into the
    /// internal free pool (preserving its heap capacity for future frames),
    /// and `output` receives the new frame's pixel data. Returns the frame
    /// index, or `None` if no frame is ready.
    ///
    /// After the initial warm-up period (≤ `buffer_count` frames), this
    /// method performs **zero heap allocations** per call.
    pub fn try_recv_into(&mut self, output: &mut Vec<u8>) -> Result<Option<u64>, ReadbackError> {
        // Priority: stashed frames from back-pressure handling.
        if let Some(mut frame) = self.stashed_frames.pop_front() {
            std::mem::swap(output, &mut frame.pixels);
            self.recycle(frame.pixels);
            return Ok(Some(frame.frame_index));
        }

        match self.receiver.try_recv() {
            Ok(ready) => {
                ready
                    .result
                    .map_err(|e| ReadbackError::MapFailed(e.to_string()))?;

                // Recycle the caller's old buffer, then write directly into it.
                let mut old = std::mem::take(output);
                old.clear();
                self.free_pool.push(old);

                let mut pixels = self.alloc_pixel_buf();
                self.extract_pixels_into(&self.buffers[ready.slot], &mut pixels);
                *output = pixels;
                self.in_flight_frames -= 1;
                Ok(Some(ready.frame_index))
            }
            Err(flume::TryRecvError::Empty | flume::TryRecvError::Disconnected) => Ok(None),
        }
    }

    // ====================================================================
    // Flush / drain
    // ====================================================================

    /// Blocks until all in-flight frames have been read back.
    ///
    /// Returns all remaining frames (stashed + in-flight) as a `Vec`.
    /// This should be called at the end of a recording session to ensure
    /// no frames are lost.
    pub fn flush(&mut self, device: &wgpu::Device) -> Result<Vec<ReadbackFrame>, ReadbackError> {
        let mut frames = Vec::new();

        // Drain stashed frames first.
        while let Some(frame) = self.stashed_frames.pop_front() {
            frames.push(frame);
        }

        // Wait for GPU to finish all pending copies.
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| ReadbackError::MapFailed(format!("device.poll failed: {e}")))?;

        // Drain the channel.
        while let Ok(ready) = self.receiver.try_recv() {
            ready
                .result
                .map_err(|e| ReadbackError::MapFailed(e.to_string()))?;

            let mut pixels = self.alloc_pixel_buf();
            self.extract_pixels_into(&self.buffers[ready.slot], &mut pixels);
            frames.push(ReadbackFrame {
                pixels,
                frame_index: ready.frame_index,
            });
            self.in_flight_frames -= 1;
        }

        Ok(frames)
    }

    // ====================================================================
    // Accessors
    // ====================================================================

    /// Returns the pixel format of the readback stream.
    #[inline]
    #[must_use]
    pub fn format(&self) -> wgpu::TextureFormat {
        self.format
    }

    /// Returns the render target dimensions.
    #[inline]
    #[must_use]
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Returns the number of bytes per pixel for the current format.
    #[inline]
    #[must_use]
    pub fn bytes_per_pixel(&self) -> u32 {
        self.bytes_per_pixel
    }

    /// Returns the number of ring-buffer slots.
    #[inline]
    #[must_use]
    pub fn buffer_count(&self) -> usize {
        self.buffers.len()
    }

    /// Returns the total number of frames submitted so far.
    #[inline]
    #[must_use]
    pub fn frames_submitted(&self) -> u64 {
        self.next_frame_index
    }

    /// Returns the expected byte size of one tightly-packed frame.
    #[inline]
    #[must_use]
    pub fn frame_byte_size(&self) -> usize {
        (self.width * self.height * self.bytes_per_pixel) as usize
    }

    // ====================================================================
    // Internal helpers
    // ====================================================================

    /// Records a texture-to-buffer copy and submits it to the GPU.
    ///
    /// The caller **must** verify that a free slot is available before
    /// calling this method.
    fn record_and_submit(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        texture: &wgpu::Texture,
    ) {
        let slot = self.write_idx;
        let frame_index = self.next_frame_index;
        let buffer = &self.buffers[slot];

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("ReadbackStream Encoder"),
        });

        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(self.padded_bytes_per_row),
                    rows_per_image: Some(self.height),
                },
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );

        queue.submit(std::iter::once(encoder.finish()));

        // Request async mapping.
        let tx = self.sender.clone();
        buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(ReadySlot {
                    slot,
                    frame_index,
                    result,
                });
            });

        self.write_idx = (self.write_idx + 1) % self.buffers.len();
        self.next_frame_index += 1;
        self.in_flight_frames += 1;
    }

    /// Extracts tightly-packed pixel data from a mapped buffer into `output`
    /// and unmaps the buffer for GPU reuse.
    fn extract_pixels_into(&self, buffer: &wgpu::Buffer, output: &mut Vec<u8>) {
        let slice = buffer.slice(..);
        let mapped = slice.get_mapped_range();

        let capacity = (self.width * self.height * self.bytes_per_pixel) as usize;
        output.clear();
        output.reserve(capacity.saturating_sub(output.capacity()));

        for row in 0..self.height {
            let start = (row * self.padded_bytes_per_row) as usize;
            let end = start + self.unpadded_bytes_per_row as usize;
            output.extend_from_slice(&mapped[start..end]);
        }

        drop(mapped);
        buffer.unmap();
    }

    /// Returns a pixel buffer from the free pool, or allocates a new one.
    fn alloc_pixel_buf(&mut self) -> Vec<u8> {
        self.free_pool.pop().unwrap_or_default()
    }

    /// Recycles a pixel buffer back into the free pool.
    fn recycle(&mut self, mut buf: Vec<u8>) {
        buf.clear();
        self.free_pool.push(buf);
    }
}
