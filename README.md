# Handwritten-Text-Recognition
Structure Recognition and Text Recognition are the major components which are required for this project.

A. Structure Recognition
Structure Recognition is required initially to figure out the structure of each handwritten character from any particular image file or document.
OCR (Optical Character Recognition) is used for recognizing the structure of each letter in the image
file. OCR basically extracts the required area which contains each alphabet which can later be used for
text recognition.

B. Text Recognition
Text Recognition is the next step in this project. Text recognition can be done by using OCR tools such as EasyOCR and Tesseract. These tools are popular and widely used for such recognition not only for text but for general recognition of structures
in a given document. The other method is to create our own model using CNN.
CNN is basically a type of neural network in which each image is converted into pixels and then masked many times over different layers which gives us a reduced yet well-defined representation
of the image. We can now pass these over a neural network which gives us a more efficient output as
compared to an ordinary neural network. The difference between the inbuilt libraries such as EasyOCR and Tesseracct, and the model we built
manually is that when we manually build it, we can design it according to our needs. For the
handwritten model, building our own model will be more accurate than EasyOCR or Tesseract.
