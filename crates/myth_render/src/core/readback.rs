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
//! natural back-pressure: if the CPU falls behind, `submit()` returns
//! `Err(ReadbackError::RingFull)` instead of silently allocating memory.
//!
//! # Usage
//!
//! ```rust,ignore
//! let mut stream = ReadbackStream::new(&device, width, height, format, 3)?;
//!
//! // Hot loop — one submit per frame, never blocks.
//! for _ in 0..100 {
//!     engine.update(dt);
//!     engine.render_active_scene();
//!     stream.submit(&device, &queue, &headless_texture)?;
//!
//!     // Opportunistically pull ready frames.
//!     while let Some(frame) = stream.try_recv()? {
//!         process(frame);
//!     }
//! }
//!
//! // Drain remaining in-flight frames (blocking).
//! stream.flush(&device, |frame| { process(frame); })?;
//! ```

use myth_core::RenderError;

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
        }
    }
}

impl std::error::Error for ReadbackError {}

impl From<ReadbackError> for myth_core::Error {
    fn from(e: ReadbackError) -> Self {
        myth_core::Error::Render(RenderError::ReadbackFailed(e.to_string()))
    }
}

/// A frame of pixel data returned by the readback stream.
pub struct ReadbackFrame {
    /// Tightly-packed pixel data (format matches the stream's texture format).
    pub pixels: Vec<u8>,
    /// Zero-based index of this frame in submission order.
    pub frame_index: u64,
}

/// High-throughput, non-blocking readback pipeline based on a ring buffer.
///
/// See [module documentation](self) for architecture details and usage.
pub struct ReadbackStream {
    buffers: Vec<wgpu::Buffer>,
    /// Next ring-buffer slot to write into.
    write_idx: usize,
    /// Monotonically increasing frame counter (incremented on each `submit`).
    next_frame_index: u64,

    // Bounded channel: sender lives in `map_async` callbacks, receiver is polled
    // by the CPU via `try_recv()` / `flush()`.
    sender: flume::Sender<ReadySlot>,
    receiver: flume::Receiver<ReadySlot>,

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
    ) -> Result<Self, ReadbackError> {
        let bytes_per_pixel = format
            .block_copy_size(None)
            .ok_or(ReadbackError::UnsupportedFormat(format))?;

        let unpadded_bytes_per_row = width * bytes_per_pixel;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_bytes_per_row = (unpadded_bytes_per_row + align - 1) / align * align;
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
            sender,
            receiver,
            width,
            height,
            format,
            bytes_per_pixel,
            unpadded_bytes_per_row,
            padded_bytes_per_row,
        })
    }

    /// Submits a non-blocking copy from `texture` to the next ring-buffer slot.
    ///
    /// The copy command is recorded and submitted immediately. When the GPU
    /// completes the copy, a `map_async` callback pushes the slot index into
    /// the internal channel, making it available via [`try_recv`](Self::try_recv).
    ///
    /// # Back-pressure
    ///
    /// If all slots are in-flight (the channel is full), this method returns
    /// [`ReadbackError::RingFull`] without submitting any work. The caller
    /// should drain frames first.
    pub fn submit(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        texture: &wgpu::Texture,
    ) -> Result<(), ReadbackError> {
        // Back-pressure: if the channel is full, all slots are in-flight.
        if self.sender.is_full() {
            return Err(ReadbackError::RingFull(self.buffers.len()));
        }

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
                // The send can only fail if the receiver has been dropped (stream
                // destroyed). In that case we silently discard.
                let _ = tx.send(ReadySlot {
                    slot,
                    frame_index,
                    result,
                });
            });

        self.write_idx = (self.write_idx + 1) % self.buffers.len();
        self.next_frame_index += 1;

        Ok(())
    }

    /// Returns the next ready frame without blocking, or `None` if no frames
    /// are available yet.
    ///
    /// After reading the pixel data, the underlying buffer is unmapped so that
    /// the GPU can reuse it.
    pub fn try_recv(&self) -> Result<Option<ReadbackFrame>, ReadbackError> {
        match self.receiver.try_recv() {
            Ok(ready) => {
                ready
                    .result
                    .map_err(|e| ReadbackError::MapFailed(e.to_string()))?;

                let frame = self.extract_pixels(&self.buffers[ready.slot], ready.frame_index);
                Ok(Some(frame))
            }
            Err(flume::TryRecvError::Empty) => Ok(None),
            Err(flume::TryRecvError::Disconnected) => Ok(None),
        }
    }

    /// Blocks until all in-flight frames have been read back, invoking
    /// `callback` for each one in submission order.
    ///
    /// This should be called at the end of a recording session to ensure no
    /// frames are lost.
    pub fn flush(&self, device: &wgpu::Device) -> Result<Vec<ReadbackFrame>, ReadbackError> {
        let mut frames = Vec::new();

        // 1. wait for all in-flight frames to become ready.
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| ReadbackError::MapFailed(format!("device.poll Wait failed: {e}")))?;

        // 2. drain the channel: at this point, the channel is filled with the last few ready signals
        while let Ok(ready) = self.receiver.try_recv() {
            // check if the async map result reported an error
            ready
                .result
                .map_err(|e| ReadbackError::MapFailed(e.to_string()))?;

            // extract pixels and push into the collection
            frames.push(self.extract_pixels(&self.buffers[ready.slot], ready.frame_index));
        }

        Ok(frames)
    }

    /// Returns the pixel format of the readback stream.
    #[inline]
    pub fn format(&self) -> wgpu::TextureFormat {
        self.format
    }

    /// Returns the render target dimensions.
    #[inline]
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Returns the number of bytes per pixel for the current format.
    #[inline]
    pub fn bytes_per_pixel(&self) -> u32 {
        self.bytes_per_pixel
    }

    /// Returns the number of ring-buffer slots.
    #[inline]
    pub fn buffer_count(&self) -> usize {
        self.buffers.len()
    }

    /// Returns the total number of frames submitted so far.
    #[inline]
    pub fn frames_submitted(&self) -> u64 {
        self.next_frame_index
    }

    /// Extracts tightly-packed pixel data from a mapped buffer and unmaps it.
    fn extract_pixels(&self, buffer: &wgpu::Buffer, frame_index: u64) -> ReadbackFrame {
        let slice = buffer.slice(..);
        let mapped = slice.get_mapped_range();

        let mut pixels =
            Vec::with_capacity((self.width * self.height * self.bytes_per_pixel) as usize);
        for row in 0..self.height {
            let start = (row * self.padded_bytes_per_row) as usize;
            let end = start + self.unpadded_bytes_per_row as usize;
            pixels.extend_from_slice(&mapped[start..end]);
        }

        drop(mapped);
        buffer.unmap();

        ReadbackFrame {
            pixels,
            frame_index,
        }
    }
}
