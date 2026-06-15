import qrcode

url = "https://kisip-demo.streamlit.app/"  # replace with your actual Streamlit URL

qr = qrcode.QRCode(
    version=1,
    error_correction=qrcode.constants.ERROR_CORRECT_H,
    box_size=10,
    border=4,
)
qr.add_data(url)
qr.make(fit=True)

img = qr.make_image(fill_color="black", back_color="white")
img.save("kisip_qr.png")
print("QR code saved as kisip_qr.png")