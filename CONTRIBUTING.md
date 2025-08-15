# Contributing to HWiNFO Analyzer

We welcome contributions to HWiNFO Analyzer! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository
2. Clone your fork locally
3. Install dependencies: `pip install -r requirements.txt`
4. Create a new branch for your feature/fix

## Development Setup

### Prerequisites
- Python 3.8 or higher
- Git

### Installation
```bash
git clone https://github.com/your-username/hwinfo-analyzer.git
cd hwinfo-analyzer
pip install -r requirements.txt
```

## Code Style

- Follow PEP 8 style guidelines
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Keep functions focused and small
- Use type hints where appropriate

## Testing

Before submitting a pull request:

1. Test your changes with different HWiNFO CSV files
2. Ensure all existing functionality still works
3. Test with both Intel and AMD systems if possible
4. Verify visualizations generate correctly

## Submitting Changes

1. Create a descriptive commit message
2. Push to your fork
3. Submit a pull request with:
   - Clear description of changes
   - Screenshots of new visualizations (if applicable)
   - Test results with sample data

## Areas for Contribution

### High Priority
- Additional CPU architecture support
- GPU vendor detection improvements
- New visualization types
- Performance optimizations
- Documentation improvements

### Medium Priority
- Additional anomaly detection algorithms
- Export formats (PDF, HTML reports)
- Configuration GUI
- Batch processing capabilities

### Low Priority
- Additional language support
- Custom threshold configuration
- Plugin architecture

## Hardware Support Contributions

When adding support for new hardware:

1. Research official thermal specifications
2. Add appropriate thresholds to `thermal_thresholds.py`
3. Update detection logic in `data_processor.py`
4. Add classification methods if needed
5. Update documentation

## Thermal Threshold Guidelines

All thermal thresholds must be based on:
- Official manufacturer specifications
- Published TjMax values
- Industry standard operating ranges
- Documented throttling temperatures

Do not use:
- Arbitrary temperature values
- Unverified sources
- Overly conservative thresholds

## Bug Reports

When reporting bugs, include:
- Python version
- Operating system
- HWiNFO version
- Sample CSV file (if possible)
- Complete error message
- Steps to reproduce

## Feature Requests

For feature requests, provide:
- Clear use case description
- Expected behavior
- Example scenarios
- Implementation suggestions (if any)

## Documentation

When contributing documentation:
- Use clear, concise language
- Include examples where helpful
- Keep technical accuracy high
- Update README.md if adding features

## Code Review Process

All contributions will be reviewed for:
- Code quality and style
- Functionality correctness
- Performance impact
- Documentation completeness
- Test coverage

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Questions?

Feel free to open an issue for any questions about contributing to the project.