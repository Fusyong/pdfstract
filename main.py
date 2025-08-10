from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
import os
import tempfile
import asyncio
from pathlib import Path
import logging

# Import conversion libraries
try:
    import pymupdf4llm
except ImportError:
    pymupdf4llm = None

try:
    from markitdown import MarkItDown
except ImportError:
    MarkItDown = None

try:
    from marker.converters.pdf import PdfConverter  # type: ignore
    from marker.models import create_model_dict  # type: ignore
    from marker.output import text_from_rendered  # type: ignore
    marker_available = True
except ImportError:
    PdfConverter = None
    create_model_dict = None
    text_from_rendered = None
    marker_available = False

try:
    from docling.document_converter import DocumentConverter
except ImportError:
    DocumentConverter = None

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    import pypdf
except ImportError:
    pypdf = None

try:
    import pypdfium2
except ImportError:
    pypdfium2 = None

try:
    import pymupdf
except ImportError:
    pymupdf = None

app = FastAPI(title="PDF to Markdown Converter", description="Convert PDF files to Markdown using various libraries")

# Create templates directory
templates = Jinja2Templates(directory="templates")

# Ensure upload directory exists
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the main HTML page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.get("/libraries")
async def get_available_libraries():
    """Get list of available conversion libraries"""
    libraries = []

    if pymupdf4llm:
        libraries.append({"name": "pymupdf4llm", "available": True})
    else:
        libraries.append({"name": "pymupdf4llm", "available": False})

    if MarkItDown:
        libraries.append({"name": "markitdown", "available": True})
    else:
        libraries.append({"name": "markitdown", "available": False})

    if marker_available:
        libraries.append({"name": "marker", "available": True})
    else:
        libraries.append({"name": "marker", "available": False})

    if DocumentConverter:
        libraries.append({"name": "docling", "available": True})
    else:
        libraries.append({"name": "docling", "available": False})

    if pdfplumber:
        libraries.append({"name": "pdfplumber", "available": True})
    else:
        libraries.append({"name": "pdfplumber", "available": False})

    if pypdf:
        libraries.append({"name": "pypdf", "available": True})
    else:
        libraries.append({"name": "pypdf", "available": False})

    if pypdfium2:
        libraries.append({"name": "pypdfium2", "available": True})
    else:
        libraries.append({"name": "pypdfium2", "available": False})

    if pymupdf:
        libraries.append({"name": "pymupdf", "available": True})
    else:
        libraries.append({"name": "pymupdf", "available": False})

    return {"libraries": libraries}

async def convert_with_pymupdf4llm(file_path: str) -> str:
    """Convert PDF using pymupdf4llm"""
    if not pymupdf4llm:
        raise HTTPException(status_code=400, detail="pymupdf4llm is not available")

    try:
        md_text = pymupdf4llm.to_markdown(file_path)
        return md_text
    except Exception as e:
        logger.error(f"Error with pymupdf4llm: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Conversion failed: {str(e)}")

async def convert_with_markitdown(file_path: str) -> str:
    """Convert PDF using markitdown"""
    if not MarkItDown:
        raise HTTPException(status_code=400, detail="markitdown is not available")

    try:
        md = MarkItDown()
        result = md.convert(file_path)
        return result.text_content
    except Exception as e:
        logger.error(f"Error with markitdown: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Conversion failed: {str(e)}")

async def convert_with_marker(file_path: str) -> str:
    """Convert PDF using marker"""
    if not marker_available:
        raise HTTPException(status_code=400, detail="marker is not available")

    try:
        # Initialize converter with models (this might take a while on first run)
        converter = PdfConverter(  # type: ignore
            artifact_dict=create_model_dict(),  # type: ignore
        )
        # Convert PDF
        rendered = converter(file_path)
        # Extract text from rendered output
        text, _, images = text_from_rendered(rendered)  # type: ignore
        return text
    except Exception as e:
        logger.error(f"Error with marker: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Conversion failed: {str(e)}")

async def convert_with_docling(file_path: str) -> str:
    """Convert PDF using docling"""
    if not DocumentConverter:
        raise HTTPException(status_code=400, detail="docling is not available")

    try:
        converter = DocumentConverter()
        result = converter.convert(file_path)
        return result.document.export_to_markdown()
    except Exception as e:
        logger.error(f"Error with docling: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Conversion failed: {str(e)}")

async def convert_with_pdfplumber(file_path: str) -> str:
    """Convert PDF using pdfplumber"""
    if not pdfplumber:
        raise HTTPException(status_code=400, detail="pdfplumber is not available")

    try:
        content = []

        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:

                # Extract text
                text = page.extract_text(layout=True)
                if text:
                    content.append(text)

                # Extract tables if any
                tables = page.extract_tables()
                for table_num, table in enumerate(tables, 1):
                    if table and any(any(cell for cell in row) for row in table):
                        content.append(f"\n### 表格 {table_num}\n")

                        # Convert table to markdown format
                        for row in table:
                            if any(cell for cell in row):
                                # Clean and format table cells
                                formatted_row = [str(cell).strip() if cell else "" for cell in row]
                                content.append("| " + " | ".join(formatted_row) + " |")

                        content.append("\n")

        return "".join(content)

    except Exception as e:
        logger.error(f"Error with pdfplumber: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Conversion failed: {str(e)}")

async def convert_with_pypdf(file_path: str) -> str:
    """Convert PDF using pypdf"""
    if not pypdf:
        raise HTTPException(status_code=400, detail="pypdf is not available")

    try:
        content = []

        with open(file_path, 'rb') as file:
            reader = pypdf.PdfReader(file)

            for page in reader.pages:

                # Extract text with layout_mode='line' to get text by lines
                # 使用layout_mode='line'来按行提取文本，这样可以保持文本的连续性
                text = page.extract_text(
                    extraction_mode="plain" # layout 可能丢东西; plain
                )
                if text:
                    content.append(text)

        return "\n".join(content)

    except Exception as e:
        logger.error(f"Error with pypdf: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Conversion failed: {str(e)}")

async def convert_with_pypdfium2(file_path: str) -> str:
    """Convert PDF using pypdfium2"""
    if not pypdfium2:
        raise HTTPException(status_code=400, detail="pypdfium2 is not available")

    try:
        content = []

        # Open PDF with pypdfium2
        pdf = pypdfium2.PdfDocument(file_path)

        for page in pdf:

            # Extract text with better text extraction method
            # 使用get_textpage().get_text_range()获取文本，然后按行处理
            textpage = page.get_textpage()
            text = textpage.get_text_range()

            content.append(text)

        pdf.close()

        return "\n".join(content)

    except Exception as e:
        logger.error(f"Error with pypdfium2: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Conversion failed: {str(e)}")

async def convert_with_pymupdf(file_path: str) -> str:
    """Convert PDF using pymupdf"""
    if not pymupdf:
        raise HTTPException(status_code=400, detail="pymupdf is not available")

    try:
        content = []

        # Open PDF with pymupdf
        doc = pymupdf.open(file_path)

        for page in doc:

            # Extract text
            text = page.get_text(sort=True)
            content.append(text)

        # Close the document
        doc.close()

        return "".join(content)

    except Exception as e:
        logger.error(f"Error with pymupdf: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Conversion failed: {str(e)}")

@app.post("/convert")
async def convert_pdf(
    file: UploadFile = File(...),
    library: str = Form(...)
):
    """Convert PDF to Markdown using specified library"""

    # Validate file type
    if not file.filename or not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
        content = await file.read()
        temp_file.write(content)
        temp_file_path = temp_file.name

    try:
        # Convert based on selected library
        if library == "pymupdf4llm":
            markdown_content = await convert_with_pymupdf4llm(temp_file_path)
        elif library == "markitdown":
            markdown_content = await convert_with_markitdown(temp_file_path)
        elif library == "marker":
            markdown_content = await convert_with_marker(temp_file_path)
        elif library == "docling":
            markdown_content = await convert_with_docling(temp_file_path)
        elif library == "pdfplumber":
            markdown_content = await convert_with_pdfplumber(temp_file_path)
        elif library == "pypdf":
            markdown_content = await convert_with_pypdf(temp_file_path)
        elif library == "pypdfium2":
            markdown_content = await convert_with_pypdfium2(temp_file_path)
        elif library == "pymupdf":
            markdown_content = await convert_with_pymupdf(temp_file_path)
        else:
            raise HTTPException(status_code=400, detail="Invalid library specified")

        return JSONResponse({
            "success": True,
            "markdown": markdown_content,
            "library_used": library,
            "filename": file.filename
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

    finally:
        # Clean up temporary file
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)