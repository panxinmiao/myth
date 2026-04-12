use crate::fxaa::FxaaSettings;
use crate::taa::TaaSettings;

/// Unified anti-aliasing mode for the rendering pipeline.
///
/// Each variant carries its own configuration payload, forming an algebraic
/// data type (ADT) that eliminates "ghost state" — only the settings for
/// the currently active technique exist in memory.
///
/// The engine automatically manages MSAA render targets, FXAA post-process
/// passes, and TAA temporal state based on the selected mode.
#[derive(Debug, Clone, Copy)]
#[allow(non_camel_case_types)]
#[derive(Default)]
pub enum AntiAliasingMode {
    /// No anti-aliasing.  Maximum performance.
    #[default]
    None,
    /// FXAA only.  Minimal overhead — smooths high-frequency noise but
    /// produces softer geometric edges.  Good for low-end / Web targets.
    FXAA(FxaaSettings),
    /// Hardware multi-sampling (e.g. 4×).  Crisp geometric edges but may
    /// exhibit PBR specular flickering.  Best for non-PBR / toon styles.
    MSAA(u32),
    /// MSAA + FXAA.  MSAA resolves geometric edges, FXAA removes specular
    /// shimmer.  Best static image quality with zero temporal ghosting.
    MSAA_FXAA(u32, FxaaSettings),
    /// Temporal Anti-Aliasing.
    /// Resolves all aliasing categories with slight temporal softening.
    TAA(TaaSettings),
    /// TAA + FXAA.  TAA handles temporal aliasing, FXAA provides extra smoothing.
    TAA_FXAA(TaaSettings, FxaaSettings),
}

impl AntiAliasingMode {
    #[inline]
    #[must_use]
    pub fn msaa_sample_count(&self) -> u32 {
        match self {
            Self::MSAA(s) | Self::MSAA_FXAA(s, _) => *s,
            _ => 1,
        }
    }

    #[inline]
    #[must_use]
    pub fn is_taa(&self) -> bool {
        matches!(self, Self::TAA(_) | Self::TAA_FXAA(..))
    }

    #[inline]
    #[must_use]
    pub fn is_fxaa(&self) -> bool {
        matches!(
            self,
            Self::FXAA(_) | Self::MSAA_FXAA(..) | Self::TAA_FXAA(..)
        )
    }

    #[inline]
    #[must_use]
    pub fn fxaa_settings(&self) -> Option<&FxaaSettings> {
        match self {
            Self::FXAA(s) | Self::MSAA_FXAA(_, s) | Self::TAA_FXAA(_, s) => Some(s),
            _ => Option::None,
        }
    }

    #[inline]
    #[must_use]
    pub fn taa_settings(&self) -> Option<&TaaSettings> {
        match self {
            Self::TAA(s) | Self::TAA_FXAA(s, _) => Some(s),
            _ => Option::None,
        }
    }

    #[must_use]
    pub fn fxaa() -> AntiAliasingMode {
        AntiAliasingMode::FXAA(FxaaSettings::default())
    }

    #[must_use]
    pub fn msaa() -> AntiAliasingMode {
        AntiAliasingMode::MSAA(4)
    }

    #[must_use]
    pub fn msaa_fxaa() -> AntiAliasingMode {
        AntiAliasingMode::MSAA_FXAA(4, FxaaSettings::default())
    }

    #[must_use]
    pub fn taa() -> AntiAliasingMode {
        AntiAliasingMode::TAA(TaaSettings::default())
    }

    #[must_use]
    pub fn taa_fxaa() -> AntiAliasingMode {
        AntiAliasingMode::TAA_FXAA(TaaSettings::default(), FxaaSettings::default())
    }
}
