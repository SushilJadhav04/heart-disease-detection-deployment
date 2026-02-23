from fpdf import FPDF

pdf = FPDF()
pdf.add_page()
pdf.set_font("Helvetica", size=12)
pdf.cell(0, 10, text="Test PDF", align='C')
pdf.output("simple_test.pdf")
print("✅ PDF Created: simple_test.pdf")