#pragma once

// Architecture-scoped file-version support for parametric models.
//
// Parametric model files (HyperWaveNet, ConcatWaveNet, ...) carry their own schema
// version namespace (major >= 1), separate from the stock NAM file version (0.x). This
// header adds an architecture-aware version check so that, at load time:
//
//   * stock architectures continue to be graded by the existing core checker
//     (is_version_supported), which rejects major >= 1 -- so a future upstream
//     WaveNet @ 1.0.0 will NOT load into this parametric build; and
//   * parametric architectures are graded against the parametric version bounds
//     below, which reject major < 1 -- so a stale HyperWaveNet exported at 0.7.0
//     will NOT load.
//
// The distinction cannot live in an IVersionSupportChecker (those see only the version
// string, and is_version_supported takes the best grade across checkers, so a checker
// can only ever widen acceptance). It must be applied where both the version and the
// architecture are in scope, i.e. at the get_dsp() load path.

#include <string>

#include "get_dsp.h"

namespace nam
{
const std::string LATEST_FULLY_SUPPORTED_PARAMETRIC_NAM_FILE_VERSION = "1.0.0";
const std::string EARLIEST_SUPPORTED_PARAMETRIC_NAM_FILE_VERSION = "1.0.0";

/// \brief Whether an architecture name uses the parametric file-version namespace.
///
/// Backed by a registry populated at static-init time via ParametricArchitectureHelper,
/// so each parametric architecture declares itself next to its config-parser registration.
bool is_parametric_architecture(const std::string& architecture);

/// \brief Grade a parametric model file version (major >= 1) against the parametric bounds.
///
/// Mirrors CoreVersionSupportChecker's grading within the 1.x namespace. Returns NO for
/// major < 1, so stock-era parametric exports (e.g. 0.7.0) are rejected.
Supported is_parametric_version_supported(const std::string& version);

/// \brief Verify a config version given its architecture.
///
/// Routes parametric architectures through is_parametric_version_supported() and all
/// others through the stock is_version_supported(). Throws std::runtime_error on NO and
/// warns on PARTIAL, matching the one-argument verify_config_version().
void verify_config_version(const std::string& version, const std::string& architecture);

/// \brief Auto-registration helper: marks an architecture name as parametric.
///
/// Create a static instance in the same translation unit as the architecture's
/// ConfigParserHelper so the two registrations share the same linkage guarantee.
struct ParametricArchitectureHelper
{
  explicit ParametricArchitectureHelper(const std::string& architecture);
};
} // namespace nam
