#include <iostream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

#include "parametric_version.h"

namespace nam
{
namespace
{
std::unordered_set<std::string>& parametric_architecture_registry()
{
  static std::unordered_set<std::string> registry;
  return registry;
}
} // namespace

bool is_parametric_architecture(const std::string& architecture)
{
  return parametric_architecture_registry().count(architecture) > 0;
}

ParametricArchitectureHelper::ParametricArchitectureHelper(const std::string& architecture)
{
  parametric_architecture_registry().insert(architecture);
}

Supported is_parametric_version_supported(const std::string& version)
{
  static const std::regex semver_regex(R"(^\d+\.\d+\.\d+$)");
  if (!std::regex_match(version, semver_regex))
    return Supported::NO;

  const Version parsed = ParseVersion(version);
  // Major 0 belongs to the stock (non-parametric) namespace; reject it here so that a
  // parametric file forged or exported with a stock-era version does not load.
  if (parsed.major < 1)
    return Supported::NO;

  const Version latest = ParseVersion(LATEST_FULLY_SUPPORTED_PARAMETRIC_NAM_FILE_VERSION);
  const Version earliest = ParseVersion(EARLIEST_SUPPORTED_PARAMETRIC_NAM_FILE_VERSION);

  if (parsed < earliest)
    return Supported::NO;
  if (parsed.major > latest.major || parsed.minor > latest.minor)
    return Supported::NO;
  if (latest < parsed)
    return Supported::PARTIAL;
  return Supported::YES;
}

void verify_config_version(const std::string& version, const std::string& architecture)
{
  const bool parametric = is_parametric_architecture(architecture);
  const Supported support = parametric ? is_parametric_version_supported(version) : is_version_supported(version);

  if (support == Supported::NO)
  {
    std::stringstream ss;
    if (parametric)
      ss << "Parametric model config for architecture '" << architecture << "' is an unsupported version " << version
         << ".";
    else
      ss << "Model config is an unsupported version " << version << ".";
    throw std::runtime_error(ss.str());
  }
  if (support == Supported::PARTIAL)
  {
    std::cerr << "Model config is a partially-supported version " << version << ". Continuing with partial support."
              << std::endl;
  }
}
} // namespace nam
