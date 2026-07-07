"""
LangChain Text Splitters - Complete Working Examples
Shows all 5 types of text splitting with real code
"""

from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
    Language,
    PythonCodeTextSplitter,
    MarkdownTextSplitter,
)

# ============================================================================
# 1. CHARACTER TEXT SPLITTER - Simplest, dumbest approach
# ============================================================================

def example_1_character_splitter():
    """Splits by exact character count. Doesn't understand meaning."""
    print("\n" + "="*70)
    print("1. CHARACTER TEXT SPLITTER")
    print("="*70)
    
    text = """
    Artificial Intelligence is transforming the world.
    Machine learning models can now understand language.
    Deep learning enables computers to learn from data.
    Neural networks mimic how our brains work.
    """
    
    splitter = CharacterTextSplitter(
        separator=" ",           # Split on spaces
        chunk_size=50,           # Each chunk max 50 characters
        chunk_overlap=10,        # Overlap of 10 chars between chunks
    )
    
    chunks = splitter.split_text(text)
    
    print(f"\nInput text length: {len(text)} characters")
    print(f"Number of chunks: {len(chunks)}")
    print("\nChunks:")
    for i, chunk in enumerate(chunks, 1):
        print(f"  Chunk {i}: {chunk[:40]}... ({len(chunk)} chars)")
    
    # ❌ PROBLEM: May cut sentences awkwardly
    print("\n⚠️  PROBLEM: Cuts at character boundaries, ignores sentence meaning")
    print("   Best for: Simple, non-critical texts only")


# ============================================================================
# 2. RECURSIVE CHARACTER SPLITTER - Smart fallback approach (MOST COMMON)
# ============================================================================

def example_2_recursive_splitter():
    """Smart splitter that tries multiple separators."""
    print("\n" + "="*70)
    print("2. RECURSIVE CHARACTER TEXT SPLITTER (Most Common!)")
    print("="*70)
    
    text = """
    Artificial Intelligence is transforming the world.
    Machine learning models can now understand language.
    
    Deep learning enables computers to learn from data.
    Neural networks mimic how our brains work.
    
    This approach keeps sentences together.
    """
    
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],  # Try these in order
        chunk_size=100,
        chunk_overlap=20,
    )
    
    chunks = splitter.split_text(text)
    
    print(f"\nInput text length: {len(text)} characters")
    print(f"Number of chunks: {len(chunks)}")
    print("\nChunks:")
    for i, chunk in enumerate(chunks, 1):
        print(f"\n  Chunk {i} ({len(chunk)} chars):")
        print(f"    {chunk[:60]}...")
    
    print("\n✅ BENEFIT: Respects paragraph breaks and sentences!")
    print("   Best for: General text, chatbots, most RAG systems")


# ============================================================================
# 3. TOKEN TEXT SPLITTER - Best for LLMs (precise token counting)
# ============================================================================

def example_3_token_splitter():
    """Splits by tokens (how LLMs actually count)."""
    print("\n" + "="*70)
    print("3. TOKEN TEXT SPLITTER (Best for LLMs)")
    print("="*70)
    
    text = """
    Artificial Intelligence is transforming the world.
    Machine learning models can now understand language.
    Deep learning enables computers to learn from data.
    """
    
    # Note: You need to have a tokenizer (requires openai, tiktoken, etc.)
    # This shows the approach with cl100k_base (OpenAI's tokenizer)
    try:
        splitter = TokenTextSplitter(
            encoding_name="cl100k_base",  # OpenAI's tokenizer
            chunk_size=30,                # 30 tokens per chunk
            chunk_overlap=5,              # 5 token overlap
        )
        
        chunks = splitter.split_text(text)
        
        print(f"\nInput text: {text[:60]}...")
        print(f"Number of tokens: ~{len(text) // 4} (rough estimate: 1 token ≈ 4 chars)")
        print(f"Number of chunks: {len(chunks)}")
        print("\nChunks (by tokens):")
        for i, chunk in enumerate(chunks, 1):
            token_estimate = len(chunk) // 4
            print(f"  Chunk {i} (~{token_estimate} tokens): {chunk[:40]}...")
        
    except ImportError:
        print("\n⚠️  Note: Requires 'pip install tiktoken'")
        print("   This splitter is the MOST ACCURATE for LLM context limits")
    
    print("\n✅ BENEFIT: Matches how LLMs count tokens!")
    print("   Best for: Precise control over model context windows (GPT, Claude, etc.)")


# ============================================================================
# 4. LANGUAGE-SPECIFIC SPLITTERS - Understands code syntax
# ============================================================================

def example_4_python_code_splitter():
    """Splits Python code by functions/classes, preserving logic."""
    print("\n" + "="*70)
    print("4. LANGUAGE-SPECIFIC SPLITTER (Python Code)")
    print("="*70)
    
    python_code = '''
def greet(name):
    """Say hello to someone."""
    return f"Hello, {name}!"

def calculate_sum(a, b):
    """Add two numbers."""
    return a + b

class Calculator:
    def __init__(self):
        self.result = 0
    
    def add(self, x):
        self.result += x
        return self
'''
    
    splitter = PythonCodeTextSplitter(
        chunk_size=50,
        chunk_overlap=10,
    )
    
    chunks = splitter.split_text(python_code)
    
    print(f"\nNumber of chunks: {len(chunks)}")
    print("Chunks (splits by functions/classes):")
    for i, chunk in enumerate(chunks, 1):
        print(f"\n  Chunk {i}:")
        lines = chunk.split('\n')
        for line in lines[:3]:  # Show first 3 lines
            print(f"    {line}")
        if len(lines) > 3:
            print(f"    ... ({len(lines)} total lines)")
    
    print("\n✅ BENEFIT: Keeps functions/classes intact!")
    print("   Best for: Code documentation, embedding code chunks")


def example_4_markdown_splitter():
    """Splits Markdown by headers, preserving structure."""
    print("\n" + "-"*70)
    print("   LANGUAGE-SPECIFIC SPLITTER (Markdown)")
    print("-"*70)
    
    markdown_text = '''
# Main Title

## Section One
This is the first section with some content.
It has multiple paragraphs.

### Subsection
More details here.

## Section Two
This is the second section.
'''
    
    splitter = MarkdownTextSplitter(
        chunk_size=100,
        chunk_overlap=20,
    )
    
    chunks = splitter.split_text(markdown_text)
    
    print(f"\nNumber of chunks: {len(chunks)}")
    print("Chunks (splits by headers):")
    for i, chunk in enumerate(chunks, 1):
        print(f"  Chunk {i}: {chunk[:50]}...")
    
    print("\n✅ BENEFIT: Respects markdown structure (headers, lists)")


# ============================================================================
# 5. SEMANTIC SPLITTER - Most intelligent but slower
# ============================================================================

def example_5_semantic_splitter():
    """Splits by semantic meaning using embeddings. Most intelligent!"""
    print("\n" + "="*70)
    print("5. SEMANTIC SPLITTER (Most Intelligent)")
    print("="*70)
    
    # Note: Semantic splitter requires embeddings model
    print("\n⚠️  Requires: pip install langchain-openai")
    print("   or pip install sentence-transformers (for local embeddings)")
    
    text = """
    Dogs are loyal animals. They form strong bonds with their owners.
    Dogs need daily exercise and play time to stay healthy.
    
    Cats are independent creatures. They prefer solitude most of the time.
    Cats are excellent at self-care and grooming themselves.
    """
    
    print("\n📊 How it works:")
    print("   1. Converts text to embeddings (vectors)")
    print("   2. Measures similarity between consecutive sentences")
    print("   3. Splits where similarity drops (topic change)")
    print("   4. Result: Semantically coherent chunks")
    
    print("\n📝 Expected output:")
    print("   Chunk 1: 'Dogs are loyal animals. They form strong bonds...'")
    print("   Chunk 2: 'Cats are independent creatures. They prefer...'")
    
    print("\n✅ BENEFIT: Groups semantically related content!")
    print("   Best for: Advanced RAG systems, high-quality results")
    print("   ⚠️  Downside: Slower and more expensive (requires embeddings)")


# ============================================================================
# PRACTICAL COMPARISON & RECOMMENDATIONS
# ============================================================================

def comparison_table():
    """Show comparison of all splitters."""
    print("\n" + "="*70)
    print("QUICK COMPARISON TABLE")
    print("="*70)
    
    comparison = {
        "Character": {"Speed": "⚡ Fast", "Quality": "❌ Poor", "Use": "Prototyping only"},
        "Recursive Char": {"Speed": "⚡ Fast", "Quality": "✅ Good", "Use": "Most projects (DEFAULT)"},
        "Token": {"Speed": "⚡ Fast", "Quality": "✅ Good", "Use": "LLM context limits"},
        "Language-Specific": {"Speed": "⚡ Fast", "Quality": "✅ Good", "Use": "Code/Markdown"},
        "Semantic": {"Speed": "🐢 Slow", "Quality": "⭐ Excellent", "Use": "Advanced RAG"},
    }
    
    print(f"\n{'Splitter':<20} {'Speed':<15} {'Quality':<15} {'Use Case':<30}")
    print("-" * 80)
    
    for splitter, info in comparison.items():
        print(f"{splitter:<20} {info['Speed']:<15} {info['Quality']:<15} {info['Use']:<30}")


# ============================================================================
# REAL-WORLD EXAMPLE: RAG System
# ============================================================================

def real_world_rag_example():
    """Show how to use splitters in a real RAG pipeline."""
    print("\n" + "="*70)
    print("REAL-WORLD EXAMPLE: RAG System")
    print("="*70)
    
    code = '''
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI

# 1. Load your document
with open("my_document.txt") as f:
    text = f.read()

# 2. Split the text (use Recursive for general text!)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # Adjust based on your model's context
    chunk_overlap=200,    # 20% overlap helps with context
)
chunks = splitter.split_text(text)

# 3. Create embeddings
embeddings = OpenAIEmbeddings()

# 4. Store in vector database
vectorstore = Chroma.from_texts(chunks, embeddings)

# 5. Query the RAG system
query = "What does the document say about X?"
results = vectorstore.similarity_search(query, k=3)
# results contains top 3 most relevant chunks!
    '''
    
    print("\n📋 Complete RAG Pipeline:")
    print(code)
    
    print("\n💡 Tips:")
    print("   • Start with RecursiveCharacterTextSplitter")
    print("   • Chunk size: 500-2000 tokens (depends on your model)")
    print("   • Overlap: 10-20% of chunk size")
    print("   • Test different sizes with your data!")


# ============================================================================
# RUN ALL EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("\n" + "🚀 LANGCHAIN TEXT SPLITTERS - COMPLETE GUIDE 🚀".center(70))
    print("Five different ways to split documents for LLM processing\n")
    
    example_1_character_splitter()
    example_2_recursive_splitter()
    example_3_token_splitter()
    example_4_python_code_splitter()
    example_4_markdown_splitter()
    example_5_semantic_splitter()
    comparison_table()
    real_world_rag_example()
    
    print("\n" + "="*70)
    print("✅ CHEAT SHEET - WHICH ONE TO USE?")
    print("="*70)
    print("""
1. 🥇 General text (chatbots, RAG, Q&A)?
   → RecursiveCharacterTextSplitter (90% of cases!)
   
2. 🥈 Need EXACT token control (GPT context limits)?
   → TokenTextSplitter
   
3. 🥉 Splitting code or markdown files?
   → PythonCodeTextSplitter or MarkdownTextSplitter
   
4. ⭐ Want highest quality, cost is not a concern?
   → SemanticTextSplitter (with embeddings)
   
5. ❌ Avoid CharacterTextSplitter (too naive!)
""")
    print("="*70)