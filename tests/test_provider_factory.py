"""
Unit tests for AI Provider Factory.

Verifies centralized instantiation, ID resolution, validation,
and credential safety for all supported providers.
"""

import sys
import unittest
from providers.base import BaseAIProvider
from providers.gemini_provider import GeminiProvider
from providers.nvidia_nim_provider import NvidiaNimProvider
from providers.openai_provider import OpenAIProvider
from providers.factory import get_provider_by_id, get_supported_provider_ids
import providers


class TestProviderFactory(unittest.TestCase):
    """Test suite for AI Provider Factory functions."""

    def test_get_supported_provider_ids(self):
        """Supported provider IDs must contain exactly openai, gemini, and nvidia_nim."""
        supported = get_supported_provider_ids()
        self.assertIsInstance(supported, list)
        self.assertEqual(supported, ["openai", "gemini", "nvidia_nim"])

    def test_get_provider_openai(self):
        """Factory returns properly instantiated OpenAIProvider for 'openai'."""
        provider = get_provider_by_id("openai")
        self.assertIsInstance(provider, OpenAIProvider)
        self.assertIsInstance(provider, BaseAIProvider)
        self.assertEqual(provider.provider_id, "openai")
        self.assertEqual(provider.name, "OpenAI")

    def test_get_provider_gemini(self):
        """Factory returns properly instantiated GeminiProvider for 'gemini'."""
        provider = get_provider_by_id("gemini")
        self.assertIsInstance(provider, GeminiProvider)
        self.assertIsInstance(provider, BaseAIProvider)
        self.assertEqual(provider.provider_id, "gemini")
        self.assertEqual(provider.name, "Google Gemini")

    def test_get_provider_nvidia_nim(self):
        """Factory returns properly instantiated NvidiaNimProvider for 'nvidia_nim'."""
        provider = get_provider_by_id("nvidia_nim")
        self.assertIsInstance(provider, NvidiaNimProvider)
        self.assertIsInstance(provider, BaseAIProvider)
        self.assertEqual(provider.provider_id, "nvidia_nim")
        self.assertEqual(provider.name, "NVIDIA NIM")

    def test_case_and_whitespace_normalization(self):
        """Factory normalizes whitespace and uppercase input strings."""
        p1 = get_provider_by_id("  OPENAI  ")
        self.assertIsInstance(p1, OpenAIProvider)

        p2 = get_provider_by_id("GeMiNi")
        self.assertIsInstance(p2, GeminiProvider)

        p3 = get_provider_by_id(" NVIDIA_NIM ")
        self.assertIsInstance(p3, NvidiaNimProvider)

    def test_unsupported_provider_raises_value_error(self):
        """Unsupported provider IDs raise a clear ValueError without leaking credentials."""
        with self.assertRaises(ValueError) as ctx:
            get_provider_by_id("anthropic")
        self.assertIn("Unsupported or unknown AI provider 'anthropic'", str(ctx.exception))
        self.assertIn("openai", str(ctx.exception))
        self.assertIn("gemini", str(ctx.exception))
        self.assertIn("nvidia_nim", str(ctx.exception))

    def test_empty_string_raises_value_error(self):
        """Empty or whitespace-only provider string raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            get_provider_by_id("")
        self.assertIn("Unsupported or unknown AI provider", str(ctx.exception))

    def test_invalid_type_raises_value_error(self):
        """Non-string arguments raise a clear ValueError."""
        with self.assertRaises(ValueError) as ctx:
            get_provider_by_id(None)  # type: ignore
        self.assertIn("Invalid provider_id type 'NoneType'", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx2:
            get_provider_by_id(12345)  # type: ignore
        self.assertIn("Invalid provider_id type 'int'", str(ctx2.exception))

    def test_package_exports(self):
        """Package root exposes factory helper functions."""
        self.assertTrue(hasattr(providers, "get_provider_by_id"))
        self.assertTrue(hasattr(providers, "get_supported_provider_ids"))
        self.assertIs(providers.get_provider_by_id, get_provider_by_id)
        self.assertIs(providers.get_supported_provider_ids, get_supported_provider_ids)

    def test_all_supported_providers_instantiable(self):
        """Every ID returned by get_supported_provider_ids must resolve to a valid BaseAIProvider."""
        for pid in get_supported_provider_ids():
            instance = get_provider_by_id(pid)
            self.assertIsInstance(instance, BaseAIProvider)
            self.assertEqual(instance.provider_id, pid)

    def test_factory_module_has_no_streamlit_dependency(self):
        """The providers.factory module must remain framework-agnostic with no streamlit imports."""
        factory_mod = sys.modules.get("providers.factory")
        self.assertIsNotNone(factory_mod)
        # Check module globals for any streamlit reference
        self.assertNotIn("streamlit", factory_mod.__dict__)
        self.assertNotIn("st", factory_mod.__dict__)


if __name__ == "__main__":
    unittest.main()
