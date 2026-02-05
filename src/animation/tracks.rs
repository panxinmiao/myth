// src/animation/track.rs
use crate::animation::values::Interpolatable;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InterpolationMode {
    Linear,
    Step,
    CubicSpline,
}

const MAX_SCAN_OFFSET: usize = 3;

#[derive(Debug, Clone, Default)]
pub struct KeyframeCursor {
    pub last_index: usize,
}

#[derive(Debug, Clone)]
pub struct KeyframeTrack<T: Interpolatable> {
    pub times: Vec<f32>,
    pub values: Vec<T>, // 如果是 CubicSpline，长度是 times.len() * 3
    pub interpolation: InterpolationMode,
}

impl<T: Interpolatable> KeyframeTrack<T> {
    pub fn new(times: Vec<f32>, values: Vec<T>, interpolation: InterpolationMode) -> Self {
        Self { times, values, interpolation }
    }

    pub fn sample(&self, time: f32) -> T {
        if self.times.is_empty() {
             // 实际上应该返回 Default 或 Panic，视你的错误处理策略而定
             // 这里为了演示简单，假设非空
             panic!("Track is empty"); 
        }

        // 1. 查找关键帧
        // partition_point 找到第一个大于 time 的索引，即 next_index
        let next_idx = self.times.partition_point(|&t| t <= time);

        self.sample_at_frame(next_idx, time)

    }


    /// 核心优化：带游标的采样
    /// cursor: 这是一个可变引用，我们会更新它
    pub fn sample_with_cursor(&self, time: f32, cursor: &mut KeyframeCursor) -> T {
        if self.times.is_empty() {
            // 简单处理空数据，实际项目中可能需要返回 Option 或 Default
             if let Some(val) = self.values.first() { return val.clone(); }
             panic!("Track is empty"); // 或者返回默认值
        }

        let len = self.times.len();
        // 快速路径：如果是静态数据（只有1帧）
        if len == 1 {
            return self.get_value_at(0).clone();
        }

        let i = cursor.last_index;

        // === O(1) 优化逻辑在这里复活 ===

        // 获取当前游标指向的时间
        // 安全检查：如果游标越界（比如切换了 Clip），重置为 0
        let t_curr = *self.times.get(i).unwrap_or(&self.times[0]);


        // 决策：向前找，还是向后找？
        let found_index = if time >= t_curr {
            // === 场景 A: 正常播放 或 快进 (Time 增加) ===
            // 我们尝试向后线性扫描 MAX_SCAN_OFFSET 步
            let mut res = None;
            // 从当前位置 i 开始，最多检查到 i + MAX_SCAN_OFFSET
            // 也就是检查区间: [i, i+1), [i+1, i+2)...
            for offset in 0..=MAX_SCAN_OFFSET {
                let idx = i + offset;
                // 边界检查：如果是最后一帧，且 time >= last_time，直接锁定最后一帧
                if idx >= len - 1 {
                    if time >= self.times[len - 1] {
                        res = Some(len - 1); // 锁定在末尾
                    }
                    break; 
                }
                
                // 检查区间 [times[idx], times[idx+1])
                // 我们已知 time >= t_curr (即 times[i])，所以只需检查右边界
                if time < self.times[idx + 1] {
                    res = Some(idx);
                    break;
                }
            }
            res
        } else {
            // === 场景 B: 倒放 或 循环重置 (Time 减少) ===
            // 我们尝试向前线性扫描
            let mut res = None;
            for offset in 0..=MAX_SCAN_OFFSET {
                // 防止下溢
                if i < offset { 
                    break; // 已经到头了，还没找到
                }
                let idx = i - offset;
                
                // 检查区间 [times[idx], times[idx+1])
                // 注意：如果 idx 是最后一个元素，这里逻辑会稍微不同，但在 "else" 分支里，
                // time < t_curr，说明 time 肯定小于 times[i]。
                // 如果 idx == i，我们知道 time < times[i]，所以它肯定不在区间 [i, i+1)
                // 所以 loop 第一次迭代 (offset=0) 其实是无效的？
                // 不一定，为了逻辑统一，我们还是检查标准区间定义。
                
                // 标准检查： time >= times[idx]
                // (右边界 time < times[idx+1] 在倒序查找中通常是满足的，因为我们是从右往左找)
                if time >= self.times[idx] {
                    // 找到了！time 在 idx 这一帧之后，且比 idx+1 小（上一轮循环已验证或隐含）
                    // 严谨起见，我们只需确认左边界
                    res = Some(idx);
                    break;
                }
            }
            res
        };

        // 更新游标逻辑
        let final_index = match found_index {
            Some(idx) => {
                // 命中缓存/局部搜索！更新游标
                cursor.last_index = idx;
                idx
            }
            None => {
                // === 场景 C: 剧烈跳变 (Scrubbing / Loop Reset) ===
                // 局部搜索失败，回退到全局二分查找 (O(log N))
                // partition_point 返回的是第一个 > time 的位置，即 "next_index"
                let next_idx = self.times.partition_point(|&t| t <= time);
                let idx = if next_idx > 0 { next_idx - 1 } else { 0 };
                
                // 更新游标，为下一次做准备
                cursor.last_index = idx;
                idx
            }
        };

        self.sample_at_frame(final_index, time)

    }


    /// 辅助方法：统一获取“值”部分
    /// 对于 Linear/Step，索引就是 index
    /// 对于 CubicSpline，值在 index * 3 + 1
    fn get_value_at(&self, index: usize) -> &T {
        match self.interpolation {
            InterpolationMode::CubicSpline => &self.values[index * 3 + 1],
            _ => &self.values[index],
        }
    }

    fn sample_at_frame(&self, index: usize, time: f32) -> T {
        let len = self.times.len();
        
        // 1. 边界情况：最后实际上没有下一帧了
        if index >= len - 1 {
            return self.get_value_at(len - 1).clone();
        }

        let next_idx = index + 1;
        let t0 = self.times[index];
        let t1 = self.times[next_idx];
        let dt = t1 - t0;

        // 防止除零
        let t = if dt > 1e-6 { (time - t0) / dt } else { 0.0 };
        // 钳制 t 在 [0, 1] 之间 (虽然理论上已经是了，但为了浮点误差安全)
        let t = t.clamp(0.0, 1.0);

        match self.interpolation {
            InterpolationMode::Step => self.get_value_at(index).clone(),
            InterpolationMode::Linear => {
                let v0 = self.get_value_at(index);
                let v1 = self.get_value_at(next_idx);
                T::interpolate_linear(v0, v1, t)
            },
            InterpolationMode::CubicSpline => {
                let i_prev = index * 3;
                let i_next = next_idx * 3;

                let v0 = &self.values[i_prev + 1];
                let out_tangent0 = &self.values[i_prev + 2];
                let in_tangent1 = &self.values[i_next];
                let v1 = &self.values[i_next + 1];

                T::interpolate_cubic(v0, out_tangent0, in_tangent1, v1, t, dt)
            }
        }
    }
    

}