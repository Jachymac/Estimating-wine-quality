# make_qr.py
import sys
import qrcode

def make_qr(url, out_file="app_qr.png", box_size=10, border=4):
	qr = qrcode.QRCode(
		version=None,
		error_correction=qrcode.constants.ERROR_CORRECT_M,
		box_size=box_size,
		border=border,
	)
	qr.add_data(url)
	qr.make(fit=True)
	img = qr.make_image(fill_color="black", back_color="white")
	img.save(out_file)


def main():
	# Accept URL as first argument or ask interactively
	if len(sys.argv) > 1:
		url = sys.argv[1]
	else:
		try:
			url = input("Enter the app URL to encode in the QR (or leave empty to abort): ").strip()
		except Exception:
			url = ""

	if not url:
		print("No URL provided. Usage: python make_qr.py <url> or run interactively and paste the URL.")
		sys.exit(1)

	out_file = "app_qr.png"
	try:
		make_qr(url, out_file=out_file)
		print(f"Saved {out_file}")
	except Exception as e:
		print("Failed to create QR:", e)


if __name__ == '__main__':
	main()