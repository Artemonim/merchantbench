"""Legacy import shim for the pre-MerchantBench SDK module name."""

try:
    from .merchantbench_tool_client import MerchantBenchToolClient
except ImportError:  # copied as two standalone files outside the sdk package
    from merchantbench_tool_client import MerchantBenchToolClient

RealShopToolClient = MerchantBenchToolClient

__all__ = ["MerchantBenchToolClient", "RealShopToolClient"]
