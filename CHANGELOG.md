# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Complete code management workflow and GitHub workflows
- Pre-commit hooks for code quality checks
- CI/CD pipeline with automated testing and building
- Contributing guidelines and Code of Conduct
- Comprehensive test suite structure
- Documentation templates for issues and pull requests

### Changed
- Modernized package configuration with pyproject.toml
- Improved Makefile with standard development commands
- Updated setup.py for backward compatibility

### Fixed
- Package build issues with lazy imports

## [0.1.0] - YYYY-MM-DD

### Added
- Initial release of VLA-Arena
- Comprehensive benchmark for Vision-Language-Action models
- Multiple task suites for evaluation
- Support for various VLA model architectures
- Data collection and conversion tools
- RLDS dataset builder

---

## How to Update This Changelog

When making changes, add entries under the `[Unreleased]` section in the appropriate category:

- **Added** for new features
- **Changed** for changes in existing functionality
- **Deprecated** for soon-to-be removed features
- **Removed** for now removed features
- **Fixed** for any bug fixes
- **Security** in case of vulnerabilities

When releasing a new version:
1. Change `[Unreleased]` to the new version number and date
2. Add a new `[Unreleased]` section at the top
3. Add comparison links at the bottom
