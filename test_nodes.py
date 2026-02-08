#!/usr/bin/env python3
"""
Test script for ComfyUI-AlvatarUtils nodes
Run this outside of ComfyUI to verify node logic works correctly.

Tests:
- MakeORM: ORM texture packing from grayscale inputs
- DebugAny: Debug output passthrough
- ConditionalExecution: Passthrough with activator logic
- ResolvePath: Path resolution (requires mocking folder_paths)

Requirements:
- PyTorch (for MakeORM tests)
- Run from the plugin directory or Docker container

Usage:
    python test_nodes.py           # Run all tests
    python test_nodes.py --skip-torch  # Skip tests requiring PyTorch
"""

import sys
import os
import tempfile

# Check for skip flags
SKIP_TORCH = "--skip-torch" in sys.argv or not os.environ.get("PYTORCH_AVAILABLE", "1") == "1"

# Check if PyTorch is available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠ PyTorch not available - some tests will be skipped")

# Add the nodes directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_make_orm():
    """Test MakeORM node with various tensor formats"""
    print("\n" + "=" * 50)
    print("Testing MakeORM")
    print("=" * 50)

    if not TORCH_AVAILABLE:
        print("⚠ PyTorch not available, skipping MakeORM tests")
        return (False, True)  # (passed=False, skipped=True)

    import torch
    import numpy as np

    # Import directly from the file to avoid folder_paths dependency
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "make_orm",
        os.path.join(os.path.dirname(__file__), "nodes", "texture", "make_orm.py")
    )
    make_orm_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(make_orm_module)
    MakeORM = make_orm_module.MakeORM

    node = MakeORM()
    tests_passed = 0
    tests_total = 0

    # Test 1: Standard 4D tensors [B, H, W, C]
    tests_total += 1
    print("\n[Test 1] Standard 4D tensors [B, H, W, C]...")
    try:
        ao = torch.rand(1, 64, 64, 3)       # RGB image
        roughness = torch.rand(1, 64, 64, 3)
        metalness = torch.rand(1, 64, 64, 3)

        result = node.make_orm(ao, roughness, metalness)
        orm = result[0]

        assert orm.shape == (1, 64, 64, 3), f"Expected (1, 64, 64, 3), got {orm.shape}"
        assert orm.dtype == torch.float32, f"Expected float32, got {orm.dtype}"
        print(f"  ✓ Output shape: {orm.shape}")
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ Failed: {e}")

    # Test 2: 3D tensors [B, H, W] (grayscale without channel dim)
    tests_total += 1
    print("\n[Test 2] 3D tensors [B, H, W] (Chord output format)...")
    try:
        ao = torch.rand(1, 64, 64)       # No channel dimension
        roughness = torch.rand(1, 64, 64)
        metalness = torch.rand(1, 64, 64)

        result = node.make_orm(ao, roughness, metalness)
        orm = result[0]

        assert orm.shape == (1, 64, 64, 3), f"Expected (1, 64, 64, 3), got {orm.shape}"
        print(f"  ✓ Output shape: {orm.shape}")
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ Failed: {e}")

    # Test 3: Mixed tensor formats
    tests_total += 1
    print("\n[Test 3] Mixed tensor formats...")
    try:
        ao = torch.rand(1, 64, 64)          # 3D
        roughness = torch.rand(1, 64, 64, 1)  # 4D single channel
        metalness = torch.rand(1, 64, 64, 3)  # 4D RGB

        result = node.make_orm(ao, roughness, metalness)
        orm = result[0]

        assert orm.shape == (1, 64, 64, 3), f"Expected (1, 64, 64, 3), got {orm.shape}"
        print(f"  ✓ Output shape: {orm.shape}")
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ Failed: {e}")

    # Test 4: Device handling (CPU tensors)
    tests_total += 1
    print("\n[Test 4] Device handling (all CPU)...")
    try:
        ao = torch.rand(1, 32, 32, 1, device='cpu')
        roughness = torch.rand(1, 32, 32, 1, device='cpu')
        metalness = torch.rand(1, 32, 32, 1, device='cpu')

        result = node.make_orm(ao, roughness, metalness)
        orm = result[0]

        assert orm.device.type == 'cpu', f"Expected cpu, got {orm.device}"
        print(f"  ✓ Output device: {orm.device}")
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ Failed: {e}")

    # Test 5: Verify channel assignment (R=AO, G=Roughness, B=Metalness)
    tests_total += 1
    print("\n[Test 5] Verify channel assignment...")
    try:
        ao = torch.ones(1, 4, 4, 1) * 0.2       # AO = 0.2
        roughness = torch.ones(1, 4, 4, 1) * 0.5  # Roughness = 0.5
        metalness = torch.ones(1, 4, 4, 1) * 0.8  # Metalness = 0.8

        result = node.make_orm(ao, roughness, metalness)
        orm = result[0]

        # Check each channel
        r_channel = orm[0, 0, 0, 0].item()
        g_channel = orm[0, 0, 0, 1].item()
        b_channel = orm[0, 0, 0, 2].item()

        assert abs(r_channel - 0.2) < 0.01, f"R channel (AO) should be 0.2, got {r_channel}"
        assert abs(g_channel - 0.5) < 0.01, f"G channel (Roughness) should be 0.5, got {g_channel}"
        assert abs(b_channel - 0.8) < 0.01, f"B channel (Metalness) should be 0.8, got {b_channel}"

        print(f"  ✓ Channels correct: R={r_channel:.2f}, G={g_channel:.2f}, B={b_channel:.2f}")
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ Failed: {e}")

    # Test 6: Larger batch size
    tests_total += 1
    print("\n[Test 6] Batch size > 1...")
    try:
        ao = torch.rand(4, 64, 64, 1)
        roughness = torch.rand(4, 64, 64, 1)
        metalness = torch.rand(4, 64, 64, 1)

        result = node.make_orm(ao, roughness, metalness)
        orm = result[0]

        assert orm.shape == (4, 64, 64, 3), f"Expected (4, 64, 64, 3), got {orm.shape}"
        print(f"  ✓ Output shape: {orm.shape}")
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ Failed: {e}")

    print(f"\nMakeORM: {tests_passed}/{tests_total} tests passed")
    return (tests_passed == tests_total, False)  # (passed, skipped)


def test_debug_any():
    """Test DebugAny node passthrough"""
    print("\n" + "=" * 50)
    print("Testing DebugAny")
    print("=" * 50)

    # Import directly from the file to avoid torch dependency from package __init__
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "debug_any",
        os.path.join(os.path.dirname(__file__), "nodes", "utils", "debug_any.py")
    )
    debug_any_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(debug_any_module)
    DebugAny = debug_any_module.DebugAny

    node = DebugAny()
    tests_passed = 0
    tests_total = 0

    # Helper to extract result from ComfyUI-style return
    def get_result(ret):
        """DebugAny returns {'ui': ..., 'result': (value, text)}"""
        if isinstance(ret, dict) and 'result' in ret:
            return ret['result']
        return ret

    # Test 1: String passthrough
    tests_total += 1
    print("\n[Test 1] String passthrough...")
    try:
        result = get_result(node.debug("test string"))
        assert result[0] == "test string", f"Expected 'test string', got {result[0]}"
        print("  ✓ String passed through correctly")
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ Failed: {e}")

    # Test 2: Integer passthrough
    tests_total += 1
    print("\n[Test 2] Integer passthrough...")
    try:
        result = get_result(node.debug(42))
        assert result[0] == 42, f"Expected 42, got {result[0]}"
        print("  ✓ Integer passed through correctly")
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ Failed: {e}")

    # Test 3: List passthrough
    tests_total += 1
    print("\n[Test 3] List passthrough...")
    try:
        test_list = [1, 2, 3, "four"]
        result = get_result(node.debug(test_list))
        assert result[0] == test_list, f"Expected {test_list}, got {result[0]}"
        print("  ✓ List passed through correctly")
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ Failed: {e}")

    # Test 4: Dict passthrough
    tests_total += 1
    print("\n[Test 4] Dict passthrough...")
    try:
        test_dict = {"key": "value", "number": 123}
        result = get_result(node.debug(test_dict))
        assert result[0] == test_dict, f"Expected {test_dict}, got {result[0]}"
        print("  ✓ Dict passed through correctly")
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ Failed: {e}")

    # Test 5: Tensor passthrough (if torch available)
    tests_total += 1
    print("\n[Test 5] Tensor passthrough...")
    try:
        import torch
        tensor = torch.rand(2, 3, 4)
        result = get_result(node.debug(tensor))
        assert torch.equal(result[0], tensor), "Tensor not passed through correctly"
        print(f"  ✓ Tensor passed through correctly (shape: {tensor.shape})")
        tests_passed += 1
    except ImportError:
        print("  ⚠ PyTorch not available, skipping tensor test")
        tests_passed += 1  # Count as pass since it's optional
    except Exception as e:
        print(f"  ✗ Failed: {e}")

    print(f"\nDebugAny: {tests_passed}/{tests_total} tests passed")
    return (tests_passed == tests_total, False)  # (passed, skipped)


def test_conditional_execution():
    """Test Continue3 synchronization barrier node passthrough"""
    print("\n" + "=" * 50)
    print("Testing Continue3 (Sync Barrier)")
    print("=" * 50)

    # Import directly from the file to avoid torch dependency from package __init__
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "continue_3",
        os.path.join(os.path.dirname(__file__), "nodes", "utils", "continue_3.py")
    )
    cond_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cond_module)
    Continue3 = cond_module.Continue3

    node = Continue3()
    tests_passed = 0
    tests_total = 0

    # Test 1: Single input passthrough
    tests_total += 1
    print("\n[Test 1] Single input passthrough (input1 only)...")
    try:
        result = node.execute(input1="test_value")
        assert result[0] == "test_value", f"Expected 'test_value', got {result[0]}"
        assert result[1] is None, f"Expected None for output2, got {result[1]}"
        assert result[2] is None, f"Expected None for output3, got {result[2]}"
        print("  ✓ input1 passed through to output1")
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ Failed: {e}")

    # Test 2: Two inputs passthrough
    tests_total += 1
    print("\n[Test 2] Two inputs passthrough...")
    try:
        result = node.execute(input1="first", input2="second")
        assert result[0] == "first", f"Expected 'first', got {result[0]}"
        assert result[1] == "second", f"Expected 'second', got {result[1]}"
        assert result[2] is None, f"Expected None for output3, got {result[2]}"
        print("  ✓ Both inputs passed through correctly")
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ Failed: {e}")

    # Test 3: All three inputs passthrough
    tests_total += 1
    print("\n[Test 3] Three inputs passthrough...")
    try:
        result = node.execute(input1="one", input2="two", input3="three")
        assert result[0] == "one", f"Expected 'one', got {result[0]}"
        assert result[1] == "two", f"Expected 'two', got {result[1]}"
        assert result[2] == "three", f"Expected 'three', got {result[2]}"
        print("  ✓ All three inputs passed through correctly")
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ Failed: {e}")

    # Test 4: Complex object passthrough
    tests_total += 1
    print("\n[Test 4] Complex object passthrough...")
    try:
        complex_obj = {"mesh": "data", "textures": [1, 2, 3]}
        result = node.execute(input1=complex_obj, input2=[1, 2, 3])
        assert result[0] == complex_obj, f"Expected {complex_obj}, got {result[0]}"
        assert result[1] == [1, 2, 3], f"Expected [1, 2, 3], got {result[1]}"
        print("  ✓ Complex objects passed through correctly")
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ Failed: {e}")

    # Test 5: No inputs (all None)
    tests_total += 1
    print("\n[Test 5] No inputs (all None)...")
    try:
        result = node.execute()
        assert result[0] is None, f"Expected None, got {result[0]}"
        assert result[1] is None, f"Expected None, got {result[1]}"
        assert result[2] is None, f"Expected None, got {result[2]}"
        print("  ✓ All outputs are None when no inputs connected")
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ Failed: {e}")

    print(f"\nContinue3: {tests_passed}/{tests_total} tests passed")
    return (tests_passed == tests_total, False)  # (passed, skipped)


def test_resolve_path():
    """Test ResolvePath node with mocked folder_paths"""
    print("\n" + "=" * 50)
    print("Testing ResolvePath")
    print("=" * 50)

    # Create temp directories to simulate ComfyUI structure
    temp_base = tempfile.mkdtemp()
    output_dir = os.path.join(temp_base, "output")
    input_dir = os.path.join(temp_base, "input")
    temp_dir = os.path.join(temp_base, "temp")

    os.makedirs(output_dir)
    os.makedirs(input_dir)
    os.makedirs(temp_dir)

    # Create test files
    output_file = os.path.join(output_dir, "test_output.glb")
    input_file = os.path.join(input_dir, "test_input.obj")
    temp_file = os.path.join(temp_dir, "test_temp.png")

    for f in [output_file, input_file, temp_file]:
        with open(f, 'w') as fp:
            fp.write("test")

    # Mock folder_paths module
    class MockFolderPaths:
        @staticmethod
        def get_output_directory():
            return output_dir

        @staticmethod
        def get_input_directory():
            return input_dir

        @staticmethod
        def get_temp_directory():
            return temp_dir

    # Import the module directly (bypassing package __init__)
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "resolve_path",
        os.path.join(os.path.dirname(__file__), "nodes", "utils", "resolve_path.py")
    )
    resolve_module = importlib.util.module_from_spec(spec)

    # Inject mock folder_paths BEFORE loading the module
    sys.modules['folder_paths'] = MockFolderPaths()
    spec.loader.exec_module(resolve_module)
    original_folder_paths = resolve_module.folder_paths
    resolve_module.folder_paths = MockFolderPaths()

    try:
        ResolvePath = resolve_module.ResolvePath

        node = ResolvePath()
        tests_passed = 0
        tests_total = 0

        # Test 1: Resolve file in output directory
        tests_total += 1
        print("\n[Test 1] Resolve file in output directory...")
        try:
            result = node.resolve("test_output.glb")
            assert result[0] == output_file, f"Expected {output_file}, got {result[0]}"
            print(f"  ✓ Resolved to: {result[0]}")
            tests_passed += 1
        except Exception as e:
            print(f"  ✗ Failed: {e}")

        # Test 2: Resolve file in input directory
        tests_total += 1
        print("\n[Test 2] Resolve file in input directory...")
        try:
            result = node.resolve("test_input.obj")
            assert result[0] == input_file, f"Expected {input_file}, got {result[0]}"
            print(f"  ✓ Resolved to: {result[0]}")
            tests_passed += 1
        except Exception as e:
            print(f"  ✗ Failed: {e}")

        # Test 3: Resolve file in temp directory
        tests_total += 1
        print("\n[Test 3] Resolve file in temp directory...")
        try:
            result = node.resolve("test_temp.png")
            assert result[0] == temp_file, f"Expected {temp_file}, got {result[0]}"
            print(f"  ✓ Resolved to: {result[0]}")
            tests_passed += 1
        except Exception as e:
            print(f"  ✗ Failed: {e}")

        # Test 4: Absolute path that exists
        tests_total += 1
        print("\n[Test 4] Absolute path that exists...")
        try:
            result = node.resolve(output_file)
            assert result[0] == output_file, f"Expected {output_file}, got {result[0]}"
            print(f"  ✓ Returned: {result[0]}")
            tests_passed += 1
        except Exception as e:
            print(f"  ✗ Failed: {e}")

        # Test 5: Non-existent file (should return original path)
        tests_total += 1
        print("\n[Test 5] Non-existent file...")
        try:
            result = node.resolve("nonexistent.xyz")
            assert result[0] == "nonexistent.xyz", f"Expected 'nonexistent.xyz', got {result[0]}"
            print(f"  ✓ Returned original: {result[0]}")
            tests_passed += 1
        except Exception as e:
            print(f"  ✗ Failed: {e}")

        print(f"\nResolvePath: {tests_passed}/{tests_total} tests passed")
        return (tests_passed == tests_total, False)  # (passed, skipped)

    finally:
        # Restore original folder_paths
        resolve_module.folder_paths = original_folder_paths

        # Cleanup temp directories
        import shutil
        shutil.rmtree(temp_base, ignore_errors=True)


def test_preprocess_image():
    """Test PrepareImageFor3D node functionality"""
    print("\n" + "=" * 50)
    print("Testing PrepareImageFor3D")
    print("=" * 50)

    if not TORCH_AVAILABLE:
        print("⚠ PyTorch not available, skipping PrepareImageFor3D tests")
        return (False, True)  # (passed=False, skipped=True)

    import torch
    import numpy as np
    from PIL import Image

    # Import directly from the file
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "prepare_image_for_3d",
        os.path.join(os.path.dirname(__file__), "nodes", "image", "prepare_image_for_3d.py")
    )
    prepare_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prepare_module)
    PrepareImageFor3D = prepare_module.PrepareImageFor3D

    node = PrepareImageFor3D()
    tests_passed = 0
    tests_total = 0

    # Helper: create test image tensor with alpha channel
    def create_rgba_tensor(h, w, has_object=True):
        """Create RGBA tensor with optional centered object"""
        img = np.zeros((h, w, 4), dtype=np.float32)
        if has_object:
            # Create a centered square object (50% of image size)
            obj_size = min(h, w) // 2
            start_y = (h - obj_size) // 2
            start_x = (w - obj_size) // 2
            img[start_y:start_y+obj_size, start_x:start_x+obj_size, :3] = 0.8  # RGB
            img[start_y:start_y+obj_size, start_x:start_x+obj_size, 3] = 1.0   # Alpha
        return torch.from_numpy(img).unsqueeze(0)  # Add batch dim

    # Test 1: Basic detection with alpha method
    tests_total += 1
    print("\n[Test 1] Alpha detection method...")
    try:
        input_tensor = create_rgba_tensor(128, 128)
        result = node.process(
            input_tensor,
            detection_method="alpha",
            margin=0.1,
            output_size=0
        )
        output = result[0]
        assert output.shape[0] == 1, f"Batch size should be 1, got {output.shape[0]}"
        print(f"  ✓ Output shape: {output.shape}")
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ Failed: {e}")

    # Test 2: Auto detection method
    tests_total += 1
    print("\n[Test 2] Auto detection method...")
    try:
        input_tensor = create_rgba_tensor(256, 256)
        result = node.process(
            input_tensor,
            detection_method="auto",
            margin=0.1,
            output_size=0
        )
        output = result[0]
        assert output.shape[0] == 1, f"Batch size should be 1, got {output.shape[0]}"
        print(f"  ✓ Output shape: {output.shape}")
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ Failed: {e}")

    # Test 3: With output_size resize
    tests_total += 1
    print("\n[Test 3] Resize output to 64x64...")
    try:
        input_tensor = create_rgba_tensor(256, 256)
        result = node.process(
            input_tensor,
            detection_method="alpha",
            margin=0.1,
            output_size=64
        )
        output = result[0]
        assert output.shape[1] == 64 and output.shape[2] == 64, \
            f"Expected 64x64, got {output.shape[1]}x{output.shape[2]}"
        print(f"  ✓ Output shape: {output.shape}")
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ Failed: {e}")

    # Test 4: Margin affects output size
    tests_total += 1
    print("\n[Test 4] Margin affects output size...")
    try:
        input_tensor = create_rgba_tensor(128, 128)
        result_small_margin = node.process(
            input_tensor,
            detection_method="alpha",
            margin=0.0,
            output_size=0
        )
        result_large_margin = node.process(
            input_tensor,
            detection_method="alpha",
            margin=0.2,
            output_size=0
        )
        h_small = result_small_margin[0].shape[1]
        h_large = result_large_margin[0].shape[1]
        # Larger margin should produce larger output (more padding)
        assert h_large >= h_small, \
            f"Larger margin should give larger output: {h_small} vs {h_large}"
        print(f"  ✓ Margin test passed: {h_small} → {h_large}")
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ Failed: {e}")

    # Test 5: Handle RGB image (no alpha channel)
    tests_total += 1
    print("\n[Test 5] Handle RGB image (no alpha channel)...")
    try:
        # Create RGB tensor (no alpha) - will use uniform_background detection
        img = np.ones((64, 64, 3), dtype=np.float32) * 0.9  # Light gray background
        img[20:44, 20:44, :] = 0.2  # Dark object in center
        input_tensor = torch.from_numpy(img).unsqueeze(0)

        result = node.process(
            input_tensor,
            detection_method="uniform_background",
            margin=0.1,
            output_size=0
        )
        output = result[0]
        print(f"  ✓ RGB input handled, output shape: {output.shape}")
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ Failed: {e}")

    # Test 6: Empty alpha (no visible pixels) - graceful handling
    tests_total += 1
    print("\n[Test 6] Handle empty alpha (no visible pixels)...")
    try:
        # Fully transparent image
        img = np.zeros((64, 64, 4), dtype=np.float32)
        input_tensor = torch.from_numpy(img).unsqueeze(0)

        result = node.process(
            input_tensor,
            detection_method="alpha",
            margin=0.1,
            output_size=0
        )
        output = result[0]
        # Should return original without crashing
        print(f"  ✓ Empty alpha handled gracefully, shape: {output.shape}")
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ Failed: {e}")

    print(f"\nPrepareImageFor3D: {tests_passed}/{tests_total} tests passed")
    return (tests_passed == tests_total, False)  # (passed, skipped)


def test_background_removal():
    """Test BackgroundRemoval node structure (inference requires ben2/transformers)"""
    print("\n" + "=" * 50)
    print("Testing BackgroundRemoval")
    print("=" * 50)

    if not TORCH_AVAILABLE:
        print("⚠ PyTorch not available, skipping BackgroundRemoval tests")
        return (False, True)  # (passed=False, skipped=True)

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "background_removal",
        os.path.join(os.path.dirname(__file__), "nodes", "image", "background_removal.py")
    )
    bg_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bg_module)
    BackgroundRemoval = bg_module.BackgroundRemoval

    tests_passed = 0
    tests_total = 0

    # Test 1: Class structure
    tests_total += 1
    print("\n[Test 1] Class structure...")
    try:
        assert hasattr(BackgroundRemoval, 'INPUT_TYPES'), "Missing INPUT_TYPES"
        assert hasattr(BackgroundRemoval, 'RETURN_TYPES'), "Missing RETURN_TYPES"
        assert hasattr(BackgroundRemoval, 'FUNCTION'), "Missing FUNCTION"
        assert hasattr(BackgroundRemoval, 'CATEGORY'), "Missing CATEGORY"
        assert BackgroundRemoval.CATEGORY == "Alvatar/Image", f"Wrong category: {BackgroundRemoval.CATEGORY}"
        print("  ✓ Class structure correct")
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ Failed: {e}")

    # Test 2: INPUT_TYPES has model dropdown
    tests_total += 1
    print("\n[Test 2] INPUT_TYPES model dropdown...")
    try:
        input_types = BackgroundRemoval.INPUT_TYPES()
        assert "required" in input_types, "Missing required inputs"
        assert "model" in input_types["required"], "Missing model input"
        model_options = input_types["required"]["model"][0]
        assert "RMBG-2.0" in model_options, "Missing RMBG-2.0 option"
        assert "BEN2" in model_options, "Missing BEN2 option"
        print(f"  ✓ Model options: {model_options}")
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ Failed: {e}")

    # Test 3: RETURN_TYPES has image and mask
    tests_total += 1
    print("\n[Test 3] RETURN_TYPES...")
    try:
        assert BackgroundRemoval.RETURN_TYPES == ("IMAGE", "MASK"), \
            f"Expected ('IMAGE', 'MASK'), got {BackgroundRemoval.RETURN_TYPES}"
        assert BackgroundRemoval.RETURN_NAMES == ("image", "mask"), \
            f"Expected ('image', 'mask'), got {BackgroundRemoval.RETURN_NAMES}"
        print("  ✓ Returns IMAGE and MASK")
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ Failed: {e}")

    # Test 4: Helper functions work
    tests_total += 1
    print("\n[Test 4] Helper functions (tensor2pil, pil2tensor)...")
    try:
        import torch
        from PIL import Image

        # Test tensor2pil
        tensor = torch.rand(1, 64, 64, 3)
        pil_img = bg_module.tensor2pil(tensor)
        assert isinstance(pil_img, Image.Image), "tensor2pil should return PIL Image"
        assert pil_img.size == (64, 64), f"Expected (64, 64), got {pil_img.size}"

        # Test pil2tensor
        tensor_back = bg_module.pil2tensor(pil_img)
        assert tensor_back.shape == (1, 64, 64, 3), f"Expected (1, 64, 64, 3), got {tensor_back.shape}"

        print("  ✓ Helper functions work correctly")
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ Failed: {e}")

    # Test 5: MODELS dict structure
    tests_total += 1
    print("\n[Test 5] MODELS configuration...")
    try:
        models = BackgroundRemoval.MODELS
        for model_name in ["RMBG-2.0", "BEN2"]:
            assert model_name in models, f"Missing {model_name}"
            assert "repo" in models[model_name], f"{model_name} missing repo"
            assert "type" in models[model_name], f"{model_name} missing type"
        print(f"  ✓ Models configured: {list(models.keys())}")
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ Failed: {e}")

    print(f"\nBackgroundRemoval: {tests_passed}/{tests_total} tests passed")
    print("  Note: Inference tests require ben2/transformers packages (run in Docker)")
    return (tests_passed == tests_total, False)  # (passed, skipped)


def test_upscale():
    """Test Upscale node structure (inference requires spandrel + model files)"""
    print("\n" + "=" * 50)
    print("Testing Upscale")
    print("=" * 50)

    if not TORCH_AVAILABLE:
        print("⚠ PyTorch not available, skipping Upscale tests")
        return (False, True)  # (passed=False, skipped=True)

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "upscale_4x",
        os.path.join(os.path.dirname(__file__), "nodes", "image", "upscale_4x.py")
    )
    upscale_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upscale_module)
    Upscale4x = upscale_module.Upscale4x

    tests_passed = 0
    tests_total = 0

    # Test 1: Class structure
    tests_total += 1
    print("\n[Test 1] Class structure...")
    try:
        assert hasattr(Upscale4x, 'INPUT_TYPES'), "Missing INPUT_TYPES"
        assert hasattr(Upscale4x, 'RETURN_TYPES'), "Missing RETURN_TYPES"
        assert hasattr(Upscale4x, 'FUNCTION'), "Missing FUNCTION"
        assert hasattr(Upscale4x, 'CATEGORY'), "Missing CATEGORY"
        assert Upscale4x.CATEGORY == "Alvatar/Image", f"Wrong category: {Upscale4x.CATEGORY}"
        print("  ✓ Class structure correct")
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ Failed: {e}")

    # Test 2: INPUT_TYPES has model dropdown
    tests_total += 1
    print("\n[Test 2] INPUT_TYPES model dropdown...")
    try:
        input_types = Upscale4x.INPUT_TYPES()
        assert "required" in input_types, "Missing required inputs"
        assert "image" in input_types["required"], "Missing image input"
        assert "model" in input_types["required"], "Missing model input"
        model_options = input_types["required"]["model"][0]
        assert "DRCT-L" in model_options, "Missing DRCT-L option"
        assert "HAT-GAN Sharp" in model_options, "Missing HAT-GAN Sharp option"
        assert "UltraSharp" in model_options, "Missing UltraSharp option"
        print(f"  ✓ Model options: {model_options}")
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ Failed: {e}")

    # Test 3: Optional parameters
    tests_total += 1
    print("\n[Test 3] Optional parameters...")
    try:
        input_types = Upscale4x.INPUT_TYPES()
        assert "optional" in input_types, "Missing optional inputs"
        assert "downsize" in input_types["optional"], "Missing downsize parameter"
        assert "tile_size" in input_types["optional"], "Missing tile_size"
        print("  ✓ Optional parameters: downsize, tile_size")
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ Failed: {e}")

    # Test 4: MODELS dict structure
    tests_total += 1
    print("\n[Test 4] MODELS configuration...")
    try:
        models = upscale_module.MODELS
        for model_name in ["DRCT-L", "HAT-GAN Sharp", "UltraSharp"]:
            assert model_name in models, f"Missing {model_name}"
            assert "filename" in models[model_name], f"{model_name} missing filename"
            assert "scale" in models[model_name], f"{model_name} missing scale"
            assert models[model_name]["scale"] == 4, f"{model_name} should be 4x scale"
        print(f"  ✓ Models configured: {list(models.keys())}")
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ Failed: {e}")

    # Test 5: Helper functions (tensor2pil, pil2tensor)
    tests_total += 1
    print("\n[Test 5] Helper functions (tensor2pil, pil2tensor)...")
    try:
        import torch
        from PIL import Image

        # Test tensor2pil
        tensor = torch.rand(1, 64, 64, 3)
        pil_img = upscale_module.tensor2pil(tensor)
        assert isinstance(pil_img, Image.Image), "tensor2pil should return PIL Image"
        assert pil_img.size == (64, 64), f"Expected (64, 64), got {pil_img.size}"

        # Test pil2tensor
        tensor_back = upscale_module.pil2tensor(pil_img)
        assert tensor_back.shape == (1, 64, 64, 3), f"Expected (1, 64, 64, 3), got {tensor_back.shape}"

        print("  ✓ Helper functions work correctly")
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ Failed: {e}")

    print(f"\nUpscale: {tests_passed}/{tests_total} tests passed")
    print("  Note: Inference tests require spandrel + model files (run in Docker)")
    return (tests_passed == tests_total, False)  # (passed, skipped)


def test_input_types():
    """Verify all nodes have proper INPUT_TYPES defined"""
    print("\n" + "=" * 50)
    print("Testing INPUT_TYPES definitions")
    print("=" * 50)

    import importlib.util
    nodes_dir = os.path.join(os.path.dirname(__file__), "nodes")

    # Load each node module individually
    node_classes = []

    # MakeORM requires torch
    if TORCH_AVAILABLE:
        spec = importlib.util.spec_from_file_location("make_orm", os.path.join(nodes_dir, "texture", "make_orm.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        node_classes.append(mod.MakeORM)
    else:
        print("⚠ Skipping MakeORM (requires PyTorch)")

    # DebugAny - no dependencies
    spec = importlib.util.spec_from_file_location("debug_any", os.path.join(nodes_dir, "utils", "debug_any.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    node_classes.append(mod.DebugAny)

    # Continue3 - no dependencies
    spec = importlib.util.spec_from_file_location("continue_3", os.path.join(nodes_dir, "utils", "continue_3.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    node_classes.append(mod.Continue3)

    # ResolvePath - needs folder_paths mock
    class MockFolderPaths:
        @staticmethod
        def get_output_directory():
            return "/tmp/output"
        @staticmethod
        def get_input_directory():
            return "/tmp/input"
        @staticmethod
        def get_temp_directory():
            return "/tmp/temp"

    sys.modules['folder_paths'] = MockFolderPaths()
    spec = importlib.util.spec_from_file_location("resolve_path", os.path.join(nodes_dir, "utils", "resolve_path.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    node_classes.append(mod.ResolvePath)

    # PrepareImageFor3D - requires torch
    if TORCH_AVAILABLE:
        spec = importlib.util.spec_from_file_location("prepare_image_for_3d", os.path.join(nodes_dir, "image", "prepare_image_for_3d.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        node_classes.append(mod.PrepareImageFor3D)
    else:
        print("⚠ Skipping PrepareImageFor3D (requires PyTorch)")

    # BackgroundRemoval - requires torch
    if TORCH_AVAILABLE:
        spec = importlib.util.spec_from_file_location("background_removal", os.path.join(nodes_dir, "image", "background_removal.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        node_classes.append(mod.BackgroundRemoval)
    else:
        print("⚠ Skipping BackgroundRemoval (requires PyTorch)")

    # Upscale4x - requires torch
    if TORCH_AVAILABLE:
        spec = importlib.util.spec_from_file_location("upscale_4x", os.path.join(nodes_dir, "image", "upscale_4x.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        node_classes.append(mod.Upscale4x)
    else:
        print("⚠ Skipping Upscale4x (requires PyTorch)")

    tests_passed = 0
    tests_total = 0

    for node_class in node_classes:
        tests_total += 1
        name = node_class.__name__
        print(f"\n[{name}] Checking INPUT_TYPES...")
        try:
            input_types = node_class.INPUT_TYPES()
            assert "required" in input_types or "optional" in input_types, \
                f"{name} missing required/optional in INPUT_TYPES"
            assert hasattr(node_class, 'RETURN_TYPES'), f"{name} missing RETURN_TYPES"
            assert hasattr(node_class, 'FUNCTION'), f"{name} missing FUNCTION"
            assert hasattr(node_class, 'CATEGORY'), f"{name} missing CATEGORY"

            func_name = node_class.FUNCTION
            assert hasattr(node_class, func_name), f"{name} missing function {func_name}"

            print(f"  ✓ {name}: RETURN_TYPES={node_class.RETURN_TYPES}, FUNCTION={func_name}")
            tests_passed += 1
        except Exception as e:
            print(f"  ✗ {name} failed: {e}")

    print(f"\nINPUT_TYPES: {tests_passed}/{tests_total} tests passed")
    return (tests_passed == tests_total, False)  # (passed, skipped)


def run_all_tests():
    """Run all tests and report results"""
    print("=" * 60)
    print("ComfyUI-AlvatarUtils Test Suite")
    print("=" * 60)

    # Results: (name, status) where status is "passed", "failed", or "skipped"
    results = []

    # Run all test functions - they return (passed: bool, skipped: bool)
    results.append(("MakeORM", test_make_orm()))
    results.append(("DebugAny", test_debug_any()))
    results.append(("Continue3", test_conditional_execution()))
    results.append(("ResolvePath", test_resolve_path()))
    results.append(("PrepareImageFor3D", test_preprocess_image()))
    results.append(("BackgroundRemoval", test_background_removal()))
    results.append(("Upscale", test_upscale()))
    results.append(("INPUT_TYPES", test_input_types()))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed_count = 0
    failed_count = 0
    skipped_count = 0

    for name, result in results:
        if isinstance(result, tuple):
            passed, skipped = result
            if skipped:
                status = "⚠ SKIPPED (requires PyTorch)"
                skipped_count += 1
            elif passed:
                status = "✓ PASSED"
                passed_count += 1
            else:
                status = "✗ FAILED"
                failed_count += 1
        else:
            # Legacy: treat bool as passed/failed, not skipped
            if result:
                status = "✓ PASSED"
                passed_count += 1
            else:
                status = "✗ FAILED"
                failed_count += 1
        print(f"  {name}: {status}")

    print("=" * 60)
    print(f"  Passed:  {passed_count}")
    print(f"  Failed:  {failed_count}")
    print(f"  Skipped: {skipped_count}")
    print("=" * 60)

    if failed_count > 0:
        print("SOME TESTS FAILED!")
        return 1
    elif skipped_count > 0 and passed_count == 0:
        print("ALL TESTS SKIPPED (install PyTorch to run)")
        return 0
    elif skipped_count > 0:
        print(f"TESTS PASSED ({skipped_count} skipped - need PyTorch)")
        return 0
    else:
        print("ALL TESTS PASSED!")
        return 0


if __name__ == "__main__":
    sys.exit(run_all_tests())
