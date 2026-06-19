from setuptools import setup, find_packages

setup(
    name="persona_support_agent",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "google-genai>=0.1.0",
        "streamlit>=1.30.0",
        "chromadb>=0.4.0",
        "langchain>=0.1.0",
        "pypdf>=3.0.0",
        "python-dotenv>=1.0.0",
        "rich>=12.0.0",
    ],
    python_requires=">=3.11",
    entry_points={
        "console_scripts": [
            "persona-support-agent=persona_support_agent.cli:main",
        ],
    },
)
