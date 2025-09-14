#!/usr/bin/env python3
"""
Phase 1 Setup Script - Foundation Setup
=======================================

This script helps you set up everything needed for Phase 1 of your
Haystack learning journey. It checks prerequisites, installs dependencies,
configures environment, and validates your setup.

Usage:
    python setup_phase1.py [--check-only] [--install-deps] [--configure-env]

Author: Haystack Learning Project
Phase: 1 - Foundation
Version: 1.0.0
"""

import os
import sys
import subprocess
import json
import platform
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import argparse

# Color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_colored(message: str, color: str = Colors.WHITE) -> None:
    """Print colored message to terminal."""
    print(f"{color}{message}{Colors.END}")

def print_header(title: str) -> None:
    """Print a formatted header."""
    print_colored(f"\n{'='*60}", Colors.CYAN)
    print_colored(f"🚀 {title}", Colors.BOLD + Colors.CYAN)
    print_colored(f"{'='*60}", Colors.CYAN)

def print_step(step: str, status: str = "INFO") -> None:
    """Print a step with status."""
    colors = {
        "INFO": Colors.BLUE,
        "SUCCESS": Colors.GREEN,
        "WARNING": Colors.YELLOW,
        "ERROR": Colors.RED
    }
    icon = {
        "INFO": "ℹ️",
        "SUCCESS": "✅",
        "WARNING": "⚠️",
        "ERROR": "❌"
    }
    print_colored(f"{icon.get(status, 'ℹ️')} {step}", colors.get(status, Colors.WHITE))

class Phase1Setup:
    """Setup manager for Phase 1 of Haystack learning."""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.phase1_dir = Path(__file__).parent
        self.setup_log = []
        self.errors = []

    def log_action(self, action: str, status: str, details: str = "") -> None:
        """Log setup actions for debugging."""
        self.setup_log.append({
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "status": status,
            "details": details
        })

    def check_python_version(self) -> bool:
        """Check if Python version meets requirements."""
        print_step("Checking Python version")

        version = sys.version_info
        required_major, required_minor = 3, 9

        if version.major >= required_major and version.minor >= required_minor:
            print_colored(f"  ✅ Python {version.major}.{version.minor}.{version.micro} (meets requirements)", Colors.GREEN)
            self.log_action("python_version_check", "success", f"{version.major}.{version.minor}.{version.micro}")
            return True
        else:
            print_colored(f"  ❌ Python {version.major}.{version.minor}.{version.micro} (requires {required_major}.{required_minor}+)", Colors.RED)
            self.log_action("python_version_check", "error", f"Need Python {required_major}.{required_minor}+")
            self.errors.append(f"Python version {required_major}.{required_minor}+ required")
            return False

    def check_virtual_environment(self) -> bool:
        """Check if running in a virtual environment."""
        print_step("Checking virtual environment")

        in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)

        if in_venv:
            venv_path = sys.prefix
            print_colored(f"  ✅ Virtual environment active: {venv_path}", Colors.GREEN)
            self.log_action("venv_check", "success", venv_path)
            return True
        else:
            print_colored("  ⚠️ Not running in virtual environment (recommended for isolation)", Colors.YELLOW)
            self.log_action("venv_check", "warning", "No virtual environment")
            return False

    def check_git_configuration(self) -> bool:
        """Check if Git is configured."""
        print_step("Checking Git configuration")

        try:
            # Check if git is available
            subprocess.run(["git", "--version"], capture_output=True, check=True)

            # Check if we're in a git repository
            result = subprocess.run(["git", "rev-parse", "--git-dir"],
                                  capture_output=True, check=True, cwd=self.project_root)
            git_dir = result.stdout.decode().strip()

            print_colored(f"  ✅ Git repository detected: {git_dir}", Colors.GREEN)
            self.log_action("git_check", "success", git_dir)
            return True

        except (subprocess.CalledProcessError, FileNotFoundError):
            print_colored("  ❌ Git not configured or not in a git repository", Colors.RED)
            self.log_action("git_check", "error", "Git not available")
            self.errors.append("Git configuration required")
            return False

    def check_required_packages(self) -> Tuple[bool, List[str]]:
        """Check if required Python packages are installed."""
        print_step("Checking required packages")

        required_packages = [
            "haystack-ai",
            "hatch",
            "pytest",
            "openai"
        ]

        missing_packages = []

        for package in required_packages:
            try:
                if package == "haystack-ai":
                    import haystack
                    print_colored(f"  ✅ {package} ({haystack.__version__})", Colors.GREEN)
                elif package == "hatch":
                    subprocess.run(["hatch", "--version"], capture_output=True, check=True)
                    print_colored(f"  ✅ {package}", Colors.GREEN)
                elif package == "pytest":
                    import pytest
                    print_colored(f"  ✅ {package} ({pytest.__version__})", Colors.GREEN)
                elif package == "openai":
                    import openai
                    print_colored(f"  ✅ {package} ({openai.__version__})", Colors.GREEN)

            except (ImportError, subprocess.CalledProcessError, FileNotFoundError):
                print_colored(f"  ❌ {package} (not installed)", Colors.RED)
                missing_packages.append(package)

        if missing_packages:
            self.log_action("package_check", "error", f"Missing: {missing_packages}")
            self.errors.extend([f"Missing package: {pkg}" for pkg in missing_packages])
            return False, missing_packages
        else:
            self.log_action("package_check", "success", "All packages available")
            return True, []

    def check_api_keys(self) -> Dict[str, bool]:
        """Check if API keys are configured."""
        print_step("Checking API key configuration")

        api_keys = {
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
            "COHERE_API_KEY": os.getenv("COHERE_API_KEY"),
            "HUGGINGFACE_API_TOKEN": os.getenv("HUGGINGFACE_API_TOKEN")
        }

        key_status = {}

        for key_name, key_value in api_keys.items():
            if key_value and len(key_value) > 10:  # Basic validation
                print_colored(f"  ✅ {key_name} configured", Colors.GREEN)
                key_status[key_name] = True
            else:
                print_colored(f"  ❌ {key_name} not configured", Colors.RED)
                key_status[key_name] = False

        # Check if at least one LLM provider is configured
        llm_providers = ["OPENAI_API_KEY", "COHERE_API_KEY"]
        has_llm_provider = any(key_status.get(key, False) for key in llm_providers)

        if not has_llm_provider:
            self.errors.append("At least one LLM provider API key required (OpenAI or Cohere)")

        self.log_action("api_key_check", "success" if has_llm_provider else "error", str(key_status))
        return key_status

    def install_dependencies(self) -> bool:
        """Install missing dependencies."""
        print_step("Installing dependencies")

        try:
            # Install from requirements.txt
            requirements_file = self.project_root / "requirements.txt"
            if requirements_file.exists():
                print_colored("  Installing from requirements.txt...", Colors.BLUE)
                subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(requirements_file)],
                             check=True)
                print_colored("  ✅ Dependencies installed successfully", Colors.GREEN)
                self.log_action("install_dependencies", "success", "From requirements.txt")
                return True
            else:
                # Install core packages individually
                core_packages = ["haystack-ai", "hatch", "pytest", "openai"]
                for package in core_packages:
                    print_colored(f"  Installing {package}...", Colors.BLUE)
                    subprocess.run([sys.executable, "-m", "pip", "install", package], check=True)

                print_colored("  ✅ Core packages installed successfully", Colors.GREEN)
                self.log_action("install_dependencies", "success", "Core packages")
                return True

        except subprocess.CalledProcessError as e:
            print_colored(f"  ❌ Failed to install dependencies: {e}", Colors.RED)
            self.log_action("install_dependencies", "error", str(e))
            self.errors.append("Dependency installation failed")
            return False

    def create_env_file(self) -> bool:
        """Create .env file from template if it doesn't exist."""
        print_step("Setting up environment file")

        env_file = self.project_root / ".env"
        env_example = self.project_root / ".env.example"

        if env_file.exists():
            print_colored("  ✅ .env file already exists", Colors.GREEN)
            return True

        if env_example.exists():
            try:
                # Copy .env.example to .env
                with open(env_example, 'r') as src, open(env_file, 'w') as dst:
                    content = src.read()
                    dst.write(content)

                print_colored("  ✅ Created .env file from template", Colors.GREEN)
                print_colored("  ⚠️ Please edit .env file to add your API keys", Colors.YELLOW)
                self.log_action("create_env_file", "success", "From template")
                return True

            except Exception as e:
                print_colored(f"  ❌ Failed to create .env file: {e}", Colors.RED)
                self.log_action("create_env_file", "error", str(e))
                return False
        else:
            print_colored("  ❌ .env.example template not found", Colors.RED)
            self.log_action("create_env_file", "error", "Template not found")
            return False

    def test_basic_functionality(self) -> bool:
        """Test basic Haystack functionality."""
        print_step("Testing basic functionality")

        try:
            # Test component creation
            from haystack import component

            @component
            class TestComponent:
                @component.output_types(result=str)
                def run(self, text: str) -> dict:
                    return {"result": f"Processed: {text}"}

            # Test component execution
            test_comp = TestComponent()
            result = test_comp.run(text="Hello Haystack")

            if result.get("result") == "Processed: Hello Haystack":
                print_colored("  ✅ Component creation and execution working", Colors.GREEN)
                self.log_action("basic_functionality_test", "success", "Component test passed")
                return True
            else:
                print_colored("  ❌ Component execution returned unexpected result", Colors.RED)
                self.log_action("basic_functionality_test", "error", "Unexpected result")
                return False

        except Exception as e:
            print_colored(f"  ❌ Basic functionality test failed: {e}", Colors.RED)
            self.log_action("basic_functionality_test", "error", str(e))
            self.errors.append("Basic functionality test failed")
            return False

    def create_phase1_workspace(self) -> bool:
        """Create Phase 1 workspace directories."""
        print_step("Creating Phase 1 workspace")

        directories = [
            self.phase1_dir / "components",
            self.phase1_dir / "pipelines",
            self.phase1_dir / "tests",
            self.phase1_dir / "notebooks",
            self.phase1_dir / "data"
        ]

        try:
            for directory in directories:
                directory.mkdir(parents=True, exist_ok=True)

                # Create __init__.py files for Python packages
                if directory.name in ["components", "pipelines", "tests"]:
                    init_file = directory / "__init__.py"
                    if not init_file.exists():
                        init_file.write_text("# Phase 1 package\n")

            print_colored("  ✅ Phase 1 workspace created", Colors.GREEN)
            self.log_action("create_workspace", "success", "Directories created")
            return True

        except Exception as e:
            print_colored(f"  ❌ Failed to create workspace: {e}", Colors.RED)
            self.log_action("create_workspace", "error", str(e))
            return False

    def generate_setup_report(self) -> None:
        """Generate setup report."""
        print_header("Setup Report")

        # System information
        print_colored("System Information:", Colors.BOLD)
        print_colored(f"  OS: {platform.system()} {platform.release()}", Colors.WHITE)
        print_colored(f"  Python: {sys.version}", Colors.WHITE)
        print_colored(f"  Working Directory: {os.getcwd()}", Colors.WHITE)

        # Setup status
        print_colored("\nSetup Status:", Colors.BOLD)
        if not self.errors:
            print_colored("  ✅ All checks passed! You're ready for Phase 1", Colors.GREEN)
        else:
            print_colored(f"  ❌ {len(self.errors)} issues found:", Colors.RED)
            for error in self.errors:
                print_colored(f"    • {error}", Colors.RED)

        # Save setup log
        log_file = self.phase1_dir / "setup_log.json"
        try:
            with open(log_file, 'w') as f:
                json.dump({
                    "setup_time": datetime.now().isoformat(),
                    "system_info": {
                        "os": f"{platform.system()} {platform.release()}",
                        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                        "working_directory": os.getcwd()
                    },
                    "setup_log": self.setup_log,
                    "errors": self.errors,
                    "success": len(self.errors) == 0
                }, f, indent=2)

            print_colored(f"\n📋 Setup log saved to: {log_file}", Colors.BLUE)

        except Exception as e:
            print_colored(f"\n⚠️ Could not save setup log: {e}", Colors.YELLOW)

    def run_full_setup(self, install_deps: bool = False, configure_env: bool = False) -> bool:
        """Run the complete setup process."""
        print_header("Phase 1 Foundation Setup")
        print_colored("Welcome to your Haystack learning journey! 🎓", Colors.BOLD + Colors.GREEN)

        success = True

        # Prerequisites check
        success &= self.check_python_version()
        self.check_virtual_environment()  # Warning only, don't fail
        success &= self.check_git_configuration()

        # Package installation
        packages_ok, missing = self.check_required_packages()
        if not packages_ok and install_deps:
            packages_ok = self.install_dependencies()
        success &= packages_ok

        # Environment setup
        if configure_env:
            self.create_env_file()

        # API keys check
        self.check_api_keys()  # Check but don't fail setup

        # Workspace setup
        success &= self.create_phase1_workspace()

        # Functionality test
        if success:
            success &= self.test_basic_functionality()

        self.generate_setup_report()

        return success


def main():
    """Main setup script entry point."""
    parser = argparse.ArgumentParser(description="Phase 1 Setup for Haystack Learning")
    parser.add_argument("--check-only", action="store_true",
                       help="Only check prerequisites without installing anything")
    parser.add_argument("--install-deps", action="store_true",
                       help="Install missing dependencies automatically")
    parser.add_argument("--configure-env", action="store_true",
                       help="Create .env file from template")
    parser.add_argument("--all", action="store_true",
                       help="Run full setup including installation and configuration")

    args = parser.parse_args()

    # Set flags for full setup
    if args.all:
        args.install_deps = True
        args.configure_env = True

    setup = Phase1Setup()

    try:
        success = setup.run_full_setup(
            install_deps=args.install_deps,
            configure_env=args.configure_env
        )

        if success:
            print_colored("\n🎉 Setup completed successfully!", Colors.GREEN + Colors.BOLD)
            print_colored("You're ready to start Phase 1 of your learning journey!", Colors.GREEN)
            print_colored("\nNext steps:", Colors.BOLD)
            print_colored("1. Review the Phase 1 README.md", Colors.WHITE)
            print_colored("2. Run: python my_first_component.py", Colors.WHITE)
            print_colored("3. Join the Haystack community on Discord", Colors.WHITE)
            print_colored("4. Start building your first components!", Colors.WHITE)
            sys.exit(0)
        else:
            print_colored("\n❌ Setup encountered issues", Colors.RED + Colors.BOLD)
            print_colored("Please resolve the errors above and run setup again", Colors.RED)
            sys.exit(1)

    except KeyboardInterrupt:
        print_colored("\n\n⚠️ Setup interrupted by user", Colors.YELLOW)
        sys.exit(1)
    except Exception as e:
        print_colored(f"\n❌ Unexpected error during setup: {e}", Colors.RED)
        sys.exit(1)


if __name__ == "__main__":
    main()
