from app.rag import read_pdf

text = read_pdf("documents/JeevithSwarup.pdf")

print(len(text))