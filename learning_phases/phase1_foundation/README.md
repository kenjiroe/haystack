# 🏗️ Phase 1: Foundation Setup
### Week 1-2: Building Your Haystack Development Foundation

[![Phase Status](https://img.shields.io/badge/Status-Ready%20to%20Start-brightgreen.svg)](./README.md)
[![Difficulty](https://img.shields.io/badge/Difficulty-Beginner-green.svg)](./README.md)
[![Duration](https://img.shields.io/badge/Duration-2%20weeks-blue.svg)](./README.md)

---

## 🎯 Phase Overview

Welcome to your Haystack learning journey! Phase 1 is all about building a solid foundation - setting up your development environment, understanding core concepts, and creating your first components and pipelines.

### What You'll Achieve
- ✅ **Working Development Environment** - Python, Haystack, and all tools configured
- ✅ **API Connections** - Connect to at least one LLM provider (OpenAI recommended)
- ✅ **Core Understanding** - Master components, pipelines, and data flow concepts
- ✅ **First Component** - Build and test your own custom component
- ✅ **Basic Pipeline** - Create a working multi-component pipeline
- ✅ **Testing Skills** - Set up and run comprehensive tests

### Learning Outcomes
By the end of Phase 1, you'll be able to:
- Explain what Haystack is and how it works
- Create custom components with proper input/output types
- Build and debug simple pipelines
- Write and run tests for your components
- Handle errors gracefully and debug issues

---

## 📅 Weekly Breakdown

### Week 1: Environment & Setup
**Goal**: Get everything working and understand the basics

#### Day 1-2: Environment Setup
- [ ] **Python Environment**: Set up Python 3.9+ with virtual environment
- [ ] **Haystack Installation**: Install and verify Haystack framework
- [ ] **IDE Configuration**: Configure VS Code or your preferred IDE
- [ ] **Git Setup**: Initialize repository and version control

#### Day 3-4: API Configuration
- [ ] **OpenAI Setup**: Create account and configure API key
- [ ] **Environment Variables**: Set up secure API key management
- [ ] **Test Connections**: Verify API connectivity with test scripts
- [ ] **Cost Monitoring**: Understand pricing and set up usage alerts

#### Day 5-7: Core Concepts
- [ ] **Component Architecture**: Understand how components work
- [ ] **Data Flow**: Learn how data moves through pipelines  
- [ ] **Type Systems**: Master input/output type definitions
- [ ] **Documentation**: Read core Haystack documentation

### Week 2: Hands-On Development
**Goal**: Build your first components and pipelines

#### Day 8-10: Component Development
- [ ] **Simple Component**: Create a text processing component
- [ ] **Type Definitions**: Add proper input/output type annotations
- [ ] **Error Handling**: Implement robust error handling
- [ ] **Unit Testing**: Write comprehensive tests

#### Day 11-12: Pipeline Creation
- [ ] **Basic Pipeline**: Connect components into a pipeline
- [ ] **Data Flow Testing**: Verify data moves correctly between components
- [ ] **Pipeline Debugging**: Learn debugging techniques and tools
- [ ] **Documentation**: Document your components and pipelines

#### Day 13-14: Integration & Review
- [ ] **Testing Framework**: Set up automated testing
- [ ] **Code Quality**: Add linting and formatting tools
- [ ] **Review & Reflection**: Assess your learning and plan Phase 2
- [ ] **Portfolio Start**: Document your first projects

---

## 🛠️ Required Tools & Setup

### System Requirements
- **Operating System**: macOS, Linux, or Windows (WSL2 recommended)
- **Python**: Version 3.9 or higher
- **RAM**: Minimum 4GB, recommended 8GB+
- **Storage**: At least 2GB free space
- **Internet**: Stable connection for API calls

### Software Installation Checklist

#### Core Development Tools
```bash
# 1. Python & Virtual Environment
python --version  # Should be 3.9+
python -m venv haystack-learning
source haystack-learning/bin/activate  # macOS/Linux
# OR
haystack-learning\Scripts\activate  # Windows

# 2. Package Management
pip install --upgrade pip
pip install hatch  # Modern Python project management

# 3. Haystack Framework
pip install haystack-ai
python -c "import haystack; print('✅ Haystack installed:', haystack.__version__)"
```

#### Development Environment
- [ ] **Visual Studio Code** with Python extension
- [ ] **Git** for version control
- [ ] **Terminal/Command Prompt** access
- [ ] **Web Browser** for documentation and testing

#### Optional but Recommended
- [ ] **Docker** for containerization (later phases)
- [ ] **Postman** for API testing
- [ ] **Jupyter Notebook** for interactive development

### API Keys Setup

#### Priority 1: OpenAI (Recommended)
```bash
# 1. Sign up at https://platform.openai.com
# 2. Create API key in dashboard
# 3. Set environment variable
export OPENAI_API_KEY="your-api-key-here"
```

#### Alternative Options
- **Cohere**: Free tier available, good for backup
- **HuggingFace**: Free access to open-source models
- **Anthropic**: High-quality models (if you have access)

---

## 🎓 Learning Resources

### Essential Reading
1. **[Haystack Documentation](https://docs.haystack.deepset.ai/)** - Official docs (start here!)
2. **[Components Guide](https://docs.haystack.deepset.ai/docs/components)** - Understanding components
3. **[Pipeline Guide](https://docs.haystack.deepset.ai/docs/pipelines)** - Building pipelines
4. **[Getting Started Tutorial](https://docs.haystack.deepset.ai/docs/quick-start)** - Hands-on tutorial

### Video Resources
- **Haystack YouTube Channel** - Official tutorials and demos
- **DeepLearning.AI Courses** - General AI/ML background
- **Python Programming Refresher** - If you need Python review

### Community Support
- **[Haystack Discord](https://discord.com/invite/VBpFzsgRVF)** - Real-time help and community
- **[GitHub Discussions](https://github.com/deepset-ai/haystack/discussions)** - Technical discussions
- **[Stack Overflow](https://stackoverflow.com/questions/tagged/haystack)** - Q&A platform

---

## 🧪 Phase 1 Projects

### Project 1: Text Analyzer Component
**Goal**: Build a component that analyzes text properties

```python
@component
class TextAnalyzer:
    """Analyzes text for various properties"""
    
    @component.output_types(
        word_count=int,
        char_count=int,
        readability_score=float,
        language=str
    )
    def run(self, text: str) -> dict:
        # Your implementation here
        pass
```

**Features to Implement**:
- Word and character counting
- Simple readability scoring
- Basic language detection
- Input validation and error handling

### Project 2: Document Preprocessor
**Goal**: Create a component that cleans and prepares text

```python
@component  
class DocumentPreprocessor:
    """Cleans and prepares documents for processing"""
    
    @component.output_types(
        clean_text=str,
        metadata=dict,
        processing_time=float
    )
    def run(self, raw_text: str, options: dict = None) -> dict:
        # Your implementation here
        pass
```

**Features to Implement**:
- Remove extra whitespace and formatting
- Handle special characters and encoding
- Extract and preserve useful metadata
- Performance monitoring

### Project 3: Basic Processing Pipeline
**Goal**: Connect your components into a working pipeline

```python
def create_text_processing_pipeline():
    """Creates a pipeline that processes and analyzes text"""
    pipeline = Pipeline()
    
    # Add your components
    pipeline.add_component("preprocessor", DocumentPreprocessor())
    pipeline.add_component("analyzer", TextAnalyzer())
    
    # Connect them
    pipeline.connect("preprocessor.clean_text", "analyzer.text")
    
    return pipeline
```

**Features to Implement**:
- End-to-end text processing
- Error handling and logging
- Performance monitoring
- Result validation

---

## ✅ Success Criteria & Assessment

### Technical Milestones
- [ ] **Environment**: Can run `python -c "import haystack; print('Success!')"` without errors
- [ ] **API Connection**: Can make successful API calls to chosen LLM provider
- [ ] **Component Creation**: Built at least 2 custom components with tests
- [ ] **Pipeline Execution**: Created and ran a multi-component pipeline
- [ ] **Testing**: Achieved >80% test coverage on your code
- [ ] **Documentation**: Clear documentation for all components and pipelines

### Knowledge Checkpoints
- [ ] Can explain the difference between components and pipelines
- [ ] Understands input/output types and data flow
- [ ] Can debug common pipeline issues
- [ ] Knows how to write and run tests
- [ ] Comfortable with error handling patterns

### Quality Gates
- [ ] **Code Quality**: All code passes linting (black, flake8)
- [ ] **Type Safety**: Proper type hints throughout
- [ ] **Error Handling**: Graceful error handling and recovery
- [ ] **Performance**: Components run efficiently on test data
- [ ] **Documentation**: Clear docstrings and usage examples

---

## 🚀 Quick Start Guide

### Day 1 - Get Started Now!

#### Step 1: Environment Setup (30 minutes)
```bash
# Clone your fork
git clone git@github.com:kenjiroe/haystack.git
cd haystack

# Set up Python environment
python -m venv haystack-learning
source haystack-learning/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### Step 2: Verify Installation (10 minutes)
```bash
# Test basic functionality
python examples/basic_example.py

# Run existing tests
python -m hatch run test:pytest test/test_telemetry.py -v
```

#### Step 3: Your First Component (1 hour)
Create `learning_phases/phase1_foundation/my_first_component.py`:

```python
from haystack import component
from typing import Dict, Any

@component
class HelloHaystack:
    """Your very first Haystack component!"""
    
    @component.output_types(greeting=str, timestamp=str)
    def run(self, name: str = "World") -> Dict[str, Any]:
        from datetime import datetime
        
        greeting = f"Hello {name} from Haystack!"
        timestamp = datetime.now().isoformat()
        
        return {
            "greeting": greeting,
            "timestamp": timestamp
        }

# Test your component
if __name__ == "__main__":
    component = HelloHaystack()
    result = component.run(name="Learner")
    print(result)
```

#### Step 4: Run and Test (15 minutes)
```bash
cd learning_phases/phase1_foundation
python my_first_component.py
```

**🎉 Congratulations!** You've just created and run your first Haystack component!

---

## 📊 Progress Tracking

### Daily Check-ins
Use this template for daily progress:

```
## Day [X] - [Date]
**Time Invested**: [Hours]
**Focus**: [What you worked on]
**Completed**: 
- [List accomplishments]
**Challenges**:
- [What was difficult]
**Tomorrow**:
- [Next steps]
```

### Weekly Review Template
```
## Week [1/2] Review
**Overall Progress**: [X%]
**Key Achievements**:
- [Major accomplishments]
**Technical Skills Gained**:
- [New skills learned]
**Challenges Overcome**:
- [Difficulties resolved]
**Next Week Focus**:
- [Areas to emphasize]
```

---

## 🆘 Troubleshooting & Support

### Common Issues & Solutions

#### Installation Problems
**Issue**: `ModuleNotFoundError: No module named 'haystack'`
**Solution**:
```bash
# Ensure virtual environment is activated
source haystack-learning/bin/activate
# Reinstall Haystack
pip install --upgrade haystack-ai
```

#### API Connection Issues
**Issue**: `Invalid API key` or connection errors
**Solution**:
```bash
# Check environment variable
echo $OPENAI_API_KEY
# Test API connection
python -c "from openai import OpenAI; client = OpenAI(); print('API working!')"
```

#### Component Creation Issues
**Issue**: Component not working or import errors
**Solution**:
- Check decorator syntax: `@component`
- Verify output types: `@component.output_types(...)`
- Ensure proper return format: `return {"key": value}`

### Getting Help
1. **Check Documentation**: Always start with official docs
2. **Search Existing Issues**: GitHub issues and Stack Overflow
3. **Ask the Community**: Discord or GitHub Discussions
4. **Create Minimal Example**: Isolate the problem
5. **Document Your Solution**: Help others who face the same issue

---

## 🎯 Ready to Begin?

### Pre-Flight Checklist
- [ ] System requirements met
- [ ] Development environment ready
- [ ] API keys configured
- [ ] Learning resources bookmarked
- [ ] Progress tracking set up
- [ ] Support channels identified

### Next Steps
1. **📋 Review**: Read through this entire document
2. **🛠️ Setup**: Complete the environment setup
3. **🏃 Start**: Begin with Day 1 tasks
4. **📝 Document**: Track your progress daily
5. **🤝 Engage**: Join the community channels

---

## 📞 Phase 1 Support

### Direct Support
- **Phase Lead**: [Your Learning Mentor]
- **Office Hours**: Fridays 3-4 PM UTC
- **Emergency Help**: GitHub Issues with `phase-1` label

### Peer Support
- **Study Group**: Join weekly Phase 1 study sessions
- **Discord Channel**: `#phase-1-foundation`
- **Progress Sharing**: Weekly show-and-tell sessions

---

**🚀 Ready to transform from AI curious to AI capable? Let's start building!**

*Remember: Every expert was once a beginner. Take it one component at a time, and you'll be amazed at what you can build!* 

[![Start Phase 1](https://img.shields.io/badge/Start-Phase%201-brightgreen.svg?style=for-the-badge)](./my_first_component.py)
[![Join Community](https://img.shields.io/badge/Join-Discord-7289da.svg?style=for-the-badge)](https://discord.com/invite/VBpFzsgRVF)

---

*Last Updated: 2024-01-15 | Next Review: Weekly*