"""
Test operators execute correctly without needing API calls.
"""

import asyncio
import sys

async def test_execution():
    """Test that operators execute correctly."""
    print("\n" + "="*70)
    print("TESTING OPERATOR EXECUTION")
    print("="*70)

    errors = []

    try:
        from langpy.core import (
            Context,
            BasePrimitive,
            Success,
            loop_while,
            map_over,
            reduce,
            when,
            branch,
            pipeline
        )

        # Create a dummy primitive that increments a counter
        class CounterPrimitive(BasePrimitive):
            async def _process(self, ctx):
                count = ctx.get("count", 0)
                return Success(ctx.set("count", count + 1))

        counter = CounterPrimitive(name="counter")

        # Test 1: loop_while
        print("\n[TEST 1: loop_while]")
        loop = loop_while(
            condition=lambda ctx: ctx.get("count", 0) < 3,
            body=counter,
            max_iterations=5
        )
        result = await loop.process(Context().set("count", 0))
        if result.is_success():
            final_count = result.unwrap().get("count")
            if final_count == 3:
                print(f"  [OK] Loop executed 3 times, count={final_count}")
            else:
                errors.append(f"  [FAIL] Expected count=3, got {final_count}")
        else:
            errors.append(f"  [FAIL] Loop failed: {result.error()}")

        # Test 2: map_over
        print("\n[TEST 2: map_over]")
        mapper = map_over(
            items=lambda ctx: ["a", "b", "c"],
            apply=counter,
            parallel=True
        )
        result = await mapper.process(Context())
        if result.is_success():
            map_results = result.unwrap().get("map_results", [])
            if len(map_results) == 3:
                print(f"  [OK] Mapped over 3 items, got {len(map_results)} results")
            else:
                errors.append(f"  [FAIL] Expected 3 results, got {len(map_results)}")
        else:
            errors.append(f"  [FAIL] Map failed: {result.error()}")

        # Test 3: reduce
        print("\n[TEST 3: reduce]")
        ctx = Context().set("val1", 10).set("val2", 20).set("val3", 30)
        reducer = reduce(
            inputs=["val1", "val2", "val3"],
            combine=lambda vals: sum(vals)
        )
        result = await reducer.process(ctx)
        if result.is_success():
            sum_result = result.unwrap().get("reduce_result")
            if sum_result == 60:
                print(f"  [OK] Reduced to sum={sum_result}")
            else:
                errors.append(f"  [FAIL] Expected sum=60, got {sum_result}")
        else:
            errors.append(f"  [FAIL] Reduce failed: {result.error()}")

        # Test 4: when (conditional)
        print("\n[TEST 4: when]")
        conditional = when(
            condition=lambda ctx: ctx.get("flag", False),
            then_do=counter,
            else_do=counter
        )
        result = await conditional.process(Context().set("flag", True).set("count", 0))
        if result.is_success():
            print(f"  [OK] Conditional executed")
        else:
            errors.append(f"  [FAIL] Conditional failed: {result.error()}")

        # Test 5: branch
        print("\n[TEST 5: branch]")
        router = branch(
            router=lambda ctx: ctx.get("route", "default"),
            routes={"a": counter, "b": counter},
            default=counter
        )
        result = await router.process(Context().set("route", "a").set("count", 0))
        if result.is_success():
            print(f"  [OK] Branch executed")
        else:
            errors.append(f"  [FAIL] Branch failed: {result.error()}")

        # Test 6: pipeline with operators
        print("\n[TEST 6: pipeline composition]")
        # Map -> Reduce pipeline
        mapper2 = map_over(
            items=lambda ctx: [1, 2, 3],
            apply=counter,
            parallel=False
        )
        reducer2 = reduce(
            inputs=lambda ctx: [r.get("count", 0) for r in ctx.get("map_results", [])],
            combine=lambda vals: sum(vals)
        )
        composed = pipeline(mapper2, reducer2)
        result = await composed.process(Context())
        if result.is_success():
            final_sum = result.unwrap().get("reduce_result")
            print(f"  [OK] Map-reduce pipeline executed, sum={final_sum}")
        else:
            errors.append(f"  [FAIL] Pipeline failed: {result.error()}")

    except Exception as e:
        errors.append(f"  [FAIL] Exception: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*70)

    if errors:
        print("ERRORS FOUND:")
        for error in errors:
            print(error)
        print("="*70)
        return False
    else:
        print("ALL EXECUTION TESTS PASSED! [OK]")
        print("="*70)
        return True


if __name__ == "__main__":
    success = asyncio.run(test_execution())
    sys.exit(0 if success else 1)
