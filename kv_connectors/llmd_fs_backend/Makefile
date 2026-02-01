PYTHON ?= python
WHEEL_DIR = wheels
DIST_DIR = dist

.PHONY: wheel build test clean wheel-gds check-gds

# Install build dependencies
deps:
	$(PYTHON) -m pip install --upgrade pip setuptools wheel build

# Check if GDS (cuFile) is available
check-gds:
	@echo "Checking for GPUDirect Storage (cuFile) support..."
	@if ldconfig -p 2>/dev/null | grep -q libcufile.so; then \
		echo "✓ cuFile library found - GDS support available"; \
		echo "  Use 'make wheel-gds' to build with GDS support"; \
	else \
		echo "✗ cuFile library not found"; \
		echo "  Default build will use CPU buffer staging only"; \
	fi

# Build wheel into dist/ and copy .whl into wheels/ (default: without GDS)
wheel: deps
	$(PYTHON) -m pip wheel . -w $(DIST_DIR)
	cp $(DIST_DIR)/llmd_fs_connector-*.whl $(WHEEL_DIR)/
	@echo ""
	@echo "Wheel created in $(WHEEL_DIR):"
	@ls -lh $(WHEEL_DIR)

# Build wheel with GDS support (requires cuFile library)
wheel-gds: deps check-gds
	@echo "Building with GDS support (ENABLE_GDS=1)..."
	ENABLE_GDS=1 $(PYTHON) -m pip wheel . -w $(DIST_DIR)
	cp $(DIST_DIR)/llmd_fs_connector-*.whl $(WHEEL_DIR)/
	@echo ""
	@echo "Wheel with GDS created in $(WHEEL_DIR):"
	@ls -lh $(WHEEL_DIR)

# Editable install for development
build:
	$(PYTHON) -m pip install -e .

# Run tests
test:
	pytest ./tests/test_fs_backend.py -v -sq

# Clean local build artifacts
clean:
	rm -rf build dist *.egg-info $(WHEEL_DIR)
	@echo "Cleaned build artifacts."
