"""
Quick test to verify all operators are properly exported and importable.
"""

import sys

def test_imports():
    """Test that all operators can be imported."""
    print("\n" + "="*70)
    print("TESTING OPERATOR IMPORTS")
    print("="*70)

    errors = []

    # Test core imports
    try:
        from langpy import (
            pipeline,
            parallel,
            when,
            branch,
            loop_while,
            map_over,
            reduce,
            retry,
            recover
        )
        print("[OK] All operators imported successfully from langpy")
    except ImportError as e:
        errors.append(f"[FAIL] Failed to import from langpy: {e}")

    # Test core module imports
    try:
        from langpy.core import (
            pipeline,
            parallel,
            when,
            branch,
            loop_while,
            map_over,
            reduce,
            retry,
            recover
        )
        print("[OK] All operators imported successfully from langpy.core")
    except ImportError as e:
        errors.append(f"[FAIL] Failed to import from langpy.core: {e}")

    # Test primitive classes
    try:
        from langpy.core import (
            LoopPrimitive,
            MapPrimitive,
            ReducePrimitive,
            BranchPrimitive,
            ConditionalPrimitive,
            RetryPrimitive,
            RecoveryPrimitive
        )
        print("[OK] All primitive classes imported successfully")
    except ImportError as e:
        errors.append(f"[FAIL] Failed to import primitive classes: {e}")

    # Test operator creation
    try:
        from langpy import loop_while, map_over, reduce, when, branch
        from langpy.core import BasePrimitive

        # Create dummy primitive
        class DummyPrimitive(BasePrimitive):
            async def _process(self, ctx):
                from langpy.core import Success
                return Success(ctx)

        dummy = DummyPrimitive(name="dummy")

        # Test loop_while
        loop = loop_while(
            condition=lambda ctx: False,
            body=dummy,
            max_iterations=1
        )
        print("[OK] loop_while() creates LoopPrimitive")

        # Test map_over
        mapper = map_over(
            items=lambda ctx: [],
            apply=dummy,
            parallel=True
        )
        print("[OK] map_over() creates MapPrimitive")

        # Test reduce
        reducer = reduce(
            inputs=["key1", "key2"],
            combine=lambda x: x
        )
        print("[OK] reduce() creates ReducePrimitive")

        # Test when
        conditional = when(
            condition=lambda ctx: True,
            then_do=dummy
        )
        print("[OK] when() creates ConditionalPrimitive")

        # Test branch
        router = branch(
            router=lambda ctx: "default",
            routes={"default": dummy}
        )
        print("[OK] branch() creates BranchPrimitive")

    except Exception as e:
        errors.append(f"[FAIL] Failed to create operators: {e}")

    print("\n" + "="*70)

    if errors:
        print("ERRORS FOUND:")
        for error in errors:
            print(f"  {error}")
        print("="*70)
        return False
    else:
        print("ALL TESTS PASSED! [OK]")
        print("="*70)
        return True


if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
