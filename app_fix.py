# app1.py
import os
import pickle
import time
import base64
import secrets
from decimal import Decimal
from datetime import datetime, timedelta
from functools import wraps
from io import BytesIO
from flask import (
    Flask, render_template, request, redirect, url_for, session, flash, abort, send_file, jsonify
)
from flask_mail import Mail, Message
import mysql.connector
from mysql.connector import pooling, Error
from werkzeug.security import generate_password_hash, check_password_hash
# Biometric libs (webcam / processing)
import cv2
import numpy as np
import face_recognition
import smtplib
import socket
from email.message import EmailMessage
from email.header import Header
from email.utils import formataddr
import string
import random

VERIFICATION_CODE_TTL = 10 * 60   # 10 minutes
VERIFICATION_MAX_ATTEMPTS = 5

# ========== REMOVE .ENV — HARD CODE CONFIG FOR LAN ==========
# ⚠️ WARNING: Do NOT use hardcoded secrets in production!
app = Flask(__name__)

# --- SECRET KEY (Hardcoded for LAN Dev) ---
app.secret_key = "your-super-secret-key-for-lan-dev-12345"  # CHANGE THIS IF YOU WANT

# --- EMAIL CONFIG (Optional — Disabled by default for LAN) ---
app.config['MAIL_SERVER'] = os.getenv('MAIL_SERVER', 'smtp.gmail.com') 
app.config['MAIL_PORT'] = int(os.getenv('MAIL_PORT', 587)) 
app.config['MAIL_USE_TLS'] = os.getenv('MAIL_USE_TLS', 'True') == 'True' 
app.config['MAIL_USE_SSL'] = os.getenv('MAIL_USE_SSL', 'False') == 'True' 
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME', 'rtobvn8191@gmail.com') 
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD', 'udxzckxybkkmqxfa') # Use app password if 2FA is enabled 
app.config['MAIL_DEFAULT_SENDER'] = os.getenv('MAIL_DEFAULT_SENDER', app.config['MAIL_USERNAME'])

# Force UTF-8 for outgoing messages
app.config['MAIL_DEFAULT_CHARSET'] = 'utf-8'

mail = None
try:
    mail = Mail(app)
    app.logger.info("Flask-Mail initialized successfully.")
except Exception as e:
    mail = None
    app.logger.warning(f"Warning: Could not initialize Flask-Mail: {e} — will fallback to smtplib.")

# --- UPLOAD FOLDER ---
UPLOAD_FOLDER = os.path.join(app.root_path, "static", "uploads", "faces")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- VERIFICATION TOKENS (In-memory) ---
MOBILE_VERIFICATION_TOKENS = {}
EMAIL_VERIFICATION_TOKENS = {}  # <-- NEW for registration
FACE_UPDATE_TOKENS = {}
WARRANTY_CLAIM_TOKENS = {}  # ✅ NEW: For face-verified warranty claims

# --- NEW: Mock payment sessions (in-memory) ---
MOCK_PAYMENT_SESSIONS = {}
MOCK_PAYMENT_TTL = 10 * 60  # 10 minutes

FACE_UPDATE_CODE_TTL = 10 * 60   # 10 minutes
FACE_UPDATE_MAX_ATTEMPTS = 3
WARRANTY_CLAIM_TTL = 10 * 60    # 10 minutes

# --- MYSQL CONFIG (Hardcoded for LAN) ---
dbconfig = {
    "host": "127.0.0.1",
    "user": "root",
    "password": "",
    "database": "ecommerce_db",
}

pool = None
def init_pool():
    global pool
    if pool is None:
        pool = pooling.MySQLConnectionPool(
            pool_name="mypool", pool_size=5, connection_timeout=6, **dbconfig
        )

def get_db():
    global pool
    if pool is None:
        init_pool()
    return pool.get_connection()

# --- DB HELPERS ---
def fetchall(query, params=None):
    conn = get_db()
    try:
        cur = conn.cursor(dictionary=True)
        cur.execute(query, params or ())
        rows = cur.fetchall()
        cur.close()
        return rows
    finally:
        conn.close()

def fetchone(query, params=None):
    rows = fetchall(query, params)
    return rows[0] if rows else None

def execute(query, params=None):
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute(query, params or ())
        conn.commit()
        last_id = cur.lastrowid
        cur.close()
        return last_id
    finally:
        conn.close()

# --- AUTH HELPERS ---
def current_user():
    uid = session.get("user_id")
    if not uid:
        return None
    return fetchone("SELECT id,name,email,is_admin,face_image,email_verified,profile_complete FROM users WHERE id=%s", (uid,))

def login_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not session.get("user_id"):
            flash("Please login to continue.", "warning")
            return redirect(url_for("login", next=request.path))
        return fn(*args, **kwargs)
    return wrapper

def admin_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        user = current_user()
        if not user or not user["is_admin"]:
            abort(403)
        return fn(*args, **kwargs)
    return wrapper

# --- CART HELPERS ---
def cart_init():
    if "cart" not in session:
        # Initialize cart as a dictionary storing { "product_id": {"qty": int, "warranty_ext": int}, ... }
        session["cart"] = {}
        session.modified = True

def get_cart_count():
    cart_init()
    # Count total quantity across all items
    total_qty = 0
    for item_data in session["cart"].values():
        # Handle old format (just quantity) or new format (dict)
        if isinstance(item_data, dict):
            total_qty += int(item_data.get("qty", 0))
        else:
            total_qty += int(item_data) # Old format
    return total_qty

def get_wishlist_count():
    user = current_user()
    if not user:
        return 0
    count = fetchone("SELECT COUNT(*) as cnt FROM wishlist WHERE user_id = %s", (user["id"],))
    return count["cnt"] if count else 0

def cart_items():
    cart_init()
    if not session["cart"]:
        return [], Decimal("0.00")

    items = []
    subtotal = Decimal("0.00")

    for pid_str, item_data in session["cart"].items():
        # Handle old format (just quantity) or new format (dict)
        if isinstance(item_data, dict):
            qty = int(item_data.get("qty", 0))
            warranty_ext = int(item_data.get("warranty_ext", 0))
        else:
            qty = int(item_data) # Old format
            warranty_ext = 0

        if qty <= 0:
            continue # Skip items with zero/negative quantity

        pid = int(pid_str)
        product = fetchone(
            f"SELECT p.*, "
            f"CASE WHEN p.image_blob IS NOT NULL THEN CONCAT('/product-image/', p.id) ELSE p.image_url END AS image_url "
            f"FROM products p WHERE p.id = %s",
            (pid,)
        )
        if product:
            line_total = Decimal(str(product["price"])) * qty
            items.append({
                "product": product,
                "qty": qty,
                "line_total": line_total,
                "warranty_extension": warranty_ext # Add extension to item object
            })
            subtotal += line_total

    return items, subtotal

# --- EMAIL HELPERS (Gracefully disabled if no credentials) ---
def send_email(to, subject, template, **kwargs):
    """
    Sends an HTML email using Flask-Mail if available; falls back to smtplib.
    Returns True on success, False on failure.
    """
    try:
        # Normalize recipients
        recipients = [to] if isinstance(to, str) else list(to or [])
        if not recipients:
            app.logger.error("send_email: no recipients provided.")
            return False

        for r in recipients:
            if not isinstance(r, str) or "@" not in r or "." not in r.split("@")[1]:
                app.logger.error(f"send_email: invalid recipient address: {r}")
                return False

        # Render template
        try:
            html_body = render_template(f"emails/{template}.html", **kwargs)
        except Exception as te:
            app.logger.exception(f"send_email: template render failed for emails/{template}.html: {te}")
            html_body = kwargs.get("body", f"<p>{subject}</p>")

        # Try Flask-Mail first
        try:
            if 'mail' in globals() and mail is not None:
                msg = Message(
                    subject=subject,
                    recipients=recipients,
                    html=html_body,
                    sender=app.config.get('MAIL_DEFAULT_SENDER') or app.config.get('MAIL_USERNAME')
                )
                msg.charset = app.config.get('MAIL_DEFAULT_CHARSET', 'utf-8')
                mail.send(msg)
                app.logger.info(f"send_email: sent via Flask-Mail to {recipients}")
                return True
        except Exception as fm_err:
            app.logger.warning(f"send_email: Flask-Mail send failed, falling back to smtplib: {fm_err}")

        # Fallback: smtplib
        smtp_server = app.config.get("MAIL_SERVER", "smtp.gmail.com")
        smtp_port = int(app.config.get("MAIL_PORT", 587))
        username = app.config.get("MAIL_USERNAME")
        password = app.config.get("MAIL_PASSWORD")
        use_tls = bool(app.config.get("MAIL_USE_TLS", True))
        use_ssl = bool(app.config.get("MAIL_USE_SSL", False))

        if not username or not password:
            app.logger.error("send_email: SMTP credentials not configured.")
            return False

        # Build EmailMessage
        email_msg = EmailMessage()
        sender_addr = app.config.get('MAIL_DEFAULT_SENDER') or username
        display_name = app.config.get('MAIL_SENDER_NAME')
        if display_name:
            email_msg['From'] = formataddr((str(Header(display_name, 'utf-8')), sender_addr))
        else:
            email_msg['From'] = sender_addr
        email_msg['To'] = ", ".join(recipients)
        email_msg['Subject'] = str(Header(subject, 'utf-8'))

        plain_text = kwargs.get("plain_text") or "This is an HTML email."
        email_msg.set_content(plain_text, subtype='plain', charset='utf-8')
        email_msg.add_alternative(html_body, subtype='html')

        # Send
        try:
            if use_ssl:
                smtp = smtplib.SMTP_SSL(smtp_server, smtp_port, timeout=15)
            else:
                smtp = smtplib.SMTP(smtp_server, smtp_port, timeout=15)
            smtp.ehlo()
            if use_tls and not use_ssl:
                smtp.starttls()
                smtp.ehlo()
            smtp.login(username, password)
            smtp.send_message(email_msg)
            smtp.quit()
            app.logger.info(f"send_email: sent via smtplib to {recipients}")
            return True
        except Exception as smtp_err:
            app.logger.exception(f"send_email: SMTP send failed: {smtp_err}")
            return False

    except Exception as e:
        app.logger.exception(f"send_email: unexpected error: {e}")
        return False

def send_mobile_verification_email(user_email, user_name, token):
    return send_email(
        to=user_email,
        subject="Mobile Face Verification - Complete Setup",
        template="mobile_verification",
        user_name=user_name,
        verification_url=url_for('mobile_verify', token=token, _external=True)
    )

@app.context_processor
def inject_now():
    user = current_user()
    return {
        "year": datetime.now().year,
        "me": user,
        "cart_count": get_cart_count(),
        "wishlist_count": get_wishlist_count() if user else 0  # ✅ Inject wishlist count
    }

# --- FACE HELPERS ---
def _save_face_image_bytes(uid, bgr_image):
    filename = f"user_{uid}.jpg"
    path = os.path.join(UPLOAD_FOLDER, filename)
    try:
        cv2.imwrite(path, bgr_image)
        rel = f"/static/uploads/faces/{filename}"
        return rel
    except Exception as e:
        app.logger.exception("Failed to write face image: %s", e)
        return None

def _decode_base64_image(data_url):
    if not data_url:
        return None, "No image data"
    if "," in data_url:
        header, encoded = data_url.split(",", 1)
    else:
        encoded = data_url

    try:
        img_bytes = base64.b64decode(encoded)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return None, "Cannot decode image"
        return img, None
    except Exception as e:
        return None, f"Decode error: {e}"

def _face_encoding_from_bgr_image(bgr_image):
    try:
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb, model="hog")
        if not boxes:
            return None, None, "No face detected"
        if len(boxes) > 1:
            areas = [(b[2]-b[0])*(b[1]-b[3]) for b in boxes]
            idx = int(np.argmax(areas))
            boxes = [boxes[idx]]

        encs = face_recognition.face_encodings(rgb, boxes)
        if not encs:
            return None, None, "No face encoding produced"
        enc = encs[0]
        enc_bytes = pickle.dumps(np.asarray(enc, dtype=np.float32))
        (top, right, bottom, left) = boxes[0]
        cropped = bgr_image[top:bottom, left:right].copy()
        if cropped.size == 0:
            cropped = bgr_image.copy() # Fallback if cropping fails

        return enc_bytes, cropped, None
    except Exception as e:
        app.logger.exception("Face encoding error: %s", e)
        return None, None, f"Face error: {e}"

# --- WEBCAM CAPTURE (SERVER-SIDE) ---
def capture_face_encoding_server(timeout_sec=20, window_title="Face verification"):
    try:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    except Exception:
        cap = cv2.VideoCapture(0)

    if not cap or not cap.isOpened():
        return None, None, "Cannot access webcam"

    start = time.time()
    encoding = None
    face_img = None
    err = None

    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_title, 640, 420)

    try:
        while time.time() - start < timeout_sec:
            ok, frame = cap.read()
            if not ok:
                err = "Failed to read from webcam"
                break

            rgb_small = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
            rgb = cv2.cvtColor(rgb_small, cv2.COLOR_BGR2RGB)
            boxes = face_recognition.face_locations(rgb, model="hog")

            cv2.putText(frame, "Align face. Press 'q' to cancel.", (10,28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

            for (t, r, b, l) in boxes:
                cv2.rectangle(frame, (l*2, t*2), (r*2, b*2), (0,200,255), 2)

            cv2.imshow(window_title, frame)

            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                err = "Cancelled by user"
                break

            if len(boxes) == 1:
                rgb_full = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                encs = face_recognition.face_encodings(rgb_full)
                if encs:
                    encoding = encs[0]
                    full_boxes = face_recognition.face_locations(rgb_full, model="hog")
                    if full_boxes:
                        (t, r, b, l) = full_boxes[0]
                        face_img = frame[t:b, l:r].copy()
                    else:
                        face_img = frame.copy() # Fallback if precise crop fails
                    break

        if encoding is None and err is None:
            err = "No single face detected (timeout)"

    finally:
        try:
            cap.release()
        except Exception:
            pass
        try:
            cv2.destroyWindow(window_title)
        except Exception:
            pass

    if encoding is None:
        return None, None, err or "Failed to capture face"

    enc_bytes = pickle.dumps(np.asarray(encoding, dtype=np.float32))
    return enc_bytes, face_img, None

def enc_distance(enc_bytes_a, enc_bytes_b):
    a = pickle.loads(enc_bytes_a)
    b = pickle.loads(enc_bytes_b)
    return float(np.linalg.norm(a - b))

# ✅ WARRANTY REMINDER HELPER FUNCTION
def send_warranty_expiration_reminders():
    """
    Sends email reminders to users whose warranties expire in 90 ± 3 days.
    Avoids duplicate emails (one per user, aggregated items).
    Returns: (sent_count, skipped_count)
    """
    try:
        # Get all relevant order_items with warranty >0 and expiration near 90 days from now
        cutoff_start = datetime.now() + timedelta(days=87)   # ~3 months (±3 days window)
        cutoff_end = datetime.now() + timedelta(days=93)

        sql = """
            SELECT 
                u.id AS user_id, u.name AS user_name, u.email AS user_email,
                p.id AS product_id, p.name AS product_name, p.warranty_years,
                o.id AS order_id, o.created_at AS order_date,
                DATE_ADD(o.created_at, INTERVAL p.warranty_years YEAR) AS warranty_end_date
            FROM users u
            JOIN orders o ON o.user_id = u.id
            JOIN order_items oi ON oi.order_id = o.id
            JOIN products p ON p.id = oi.product_id
            WHERE 
                p.warranty_years > 0
                AND o.created_at IS NOT NULL
                AND DATE_ADD(o.created_at, INTERVAL p.warranty_years YEAR) BETWEEN %s AND %s
            ORDER BY u.id, warranty_end_date
        """
        rows = fetchall(sql, (cutoff_start, cutoff_end))

        from collections import defaultdict
        user_items = defaultdict(list)
        for row in rows:
            user_items[row["user_id"]].append({
                "product_name": row["product_name"],
                "order_id": row["order_id"],
                "warranty_end_date": row["warranty_end_date"]
            })

        sent = 0
        skipped = 0
        for user_id, items in user_items.items():
            user_row = next((r for r in rows if r["user_id"] == user_id), None)
            if not user_row:
                continue

            success = send_email(
                to=user_row["user_email"],
                subject="[Action Required] Warranty Expiring for Your Products",
                template="warranty_expiring",
                user_name=user_row["user_name"],
                expiring_items=items,
                shop_name="shop ease"
            )
            if success:
                sent += 1
                app.logger.info(f"Warranty reminder sent to {user_row['user_email']} ({len(items)} items)")
            else:
                skipped += 1
                app.logger.warning(f"Failed to send warranty reminder to {user_row['user_email']}")

        return sent, skipped
    except Exception as e:
        app.logger.exception("Error in send_warranty_expiration_reminders(): %s", e)
        return 0, -1


@app.route("/admin")
@admin_required
def admin_dashboard():
    products = fetchall(
        "SELECT p.*, "
        "CASE WHEN p.image_blob IS NOT NULL THEN CONCAT('/product-image/', p.id) ELSE p.image_url END AS image_url, "
        "c.name AS category_name "
        "FROM products p JOIN categories c ON c.id=p.category_id ORDER BY p.created_at DESC"
    )
    cats = fetchall("SELECT * FROM categories ORDER BY name")
    return render_template("admin_dashboard.html", products=products, categories=cats)

@app.route("/admin/product/new", methods=["POST"])
@admin_required
def admin_product_new():
    name = request.form.get("name", "").strip()
    description = request.form.get("description", "").strip()
    price = Decimal(request.form.get("price", "0") or "0")
    warranty_years = int(request.form.get("warranty_years", 0) or 0)
    category_id = int(request.form.get("category_id"))
    stock = int(request.form.get("stock", "100") or "100")
    image_url = (request.form.get("image_url") or "").strip()
    file = request.files.get("image")

    image_blob = None
    image_mimetype = None
    if file and file.filename:
        data = file.read()
        if len(data) > 8 * 1024 * 1024: # 8MB limit
            flash("Image too large (max 8MB).", "danger")
            return redirect(url_for("admin_dashboard"))
        image_blob = data
        image_mimetype = file.mimetype or "image/jpeg"

    if image_blob:
        execute(
            "INSERT INTO products (name, description, price, warranty_years, image_blob, image_mimetype, image_url, category_id, stock) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (name, description, str(price), warranty_years, image_blob, image_mimetype, None, category_id, stock),
        )
    else:
        execute(
            "INSERT INTO products (name, description, price, warranty_years, image_url, category_id, stock) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s)",
            (name, description, str(price), warranty_years, image_url, category_id, stock),
        )

    flash("Product created.", "success")
    return redirect(url_for("admin_dashboard"))

@app.route("/admin/product/<int:pid>/delete", methods=["POST"])
@admin_required
def admin_product_delete(pid):
    execute("DELETE FROM products WHERE id=%s", (pid,))
    flash("Product deleted.", "warning")
    return redirect(url_for("admin_dashboard"))

# --- EMAIL VERIFICATION ROUTES (unchanged) ---
@app.route("/verify-email/<token>")
def verify_email(token):
    if token not in EMAIL_VERIFICATION_TOKENS:
        flash("Invalid or expired verification link.", "danger")
        return redirect(url_for("register"))

    data = EMAIL_VERIFICATION_TOKENS[token]
    if time.time() - data["timestamp"] > 3600: # 1 hour TTL
        del EMAIL_VERIFICATION_TOKENS[token]
        flash("Verification link expired. Please register again.", "warning")
        return redirect(url_for("register"))

    try:
        uid = execute(
            "INSERT INTO users (name, email, password_hash, email_verified, profile_complete) VALUES (%s,%s,%s,1,0)",
            (data["name"], data["email"], data["password_hash"])
        )
        session["user_id"] = uid  # <-- FIX: Set user_id, not pending_profile_user_id
        session.modified = True
        del EMAIL_VERIFICATION_TOKENS[token] # Clean up after successful creation
        flash("Email verified! Please complete your profile.", "success")
        return redirect(url_for("complete_profile"))
    except Exception as e:
        app.logger.error(f"User creation failed after email verification: {e}")
        flash("An error occurred during registration. Please try again.", "danger")
        return redirect(url_for("register"))

# --- ROUTES (index, categories, product image, search, etc.) ---
@app.route("/product-image/<int:pid>")
def product_image(pid):
    row = fetchone("SELECT image_blob, image_mimetype FROM products WHERE id=%s", (pid,))
    if not row or not row["image_blob"]:
        abort(404)
    blob = row["image_blob"]
    mimetype = row.get("image_mimetype") or "image/jpeg"
    return send_file(BytesIO(blob), mimetype=mimetype)

@app.route("/api/generate-verification-token", methods=["POST"])
def generate_verification_token():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "No JSON data received"}), 400

        email = data.get("email")
        if not email:
            return jsonify({"success": False, "error": "Email is required"}), 400
        if "@" not in email or "." not in email.split("@")[1]:
            return jsonify({"success": False, "error": "Invalid email format"}), 400

        user = fetchone("SELECT * FROM users WHERE email=%s", (email,))
        if not user:
            return jsonify({"success": False, "error": "User not found"}), 404

        token = secrets.token_urlsafe(32)
        verification_data = {
            "email": email,
            "user_id": user["id"],
            "timestamp": time.time(),
            "status": "pending"
        }
        MOBILE_VERIFICATION_TOKENS[token] = verification_data

        if send_mobile_verification_email(email, user["name"], token):
            return jsonify({
                "success": True, 
                "token": token,
                "verification_url": url_for('mobile_verify', token=token, _external=True)
            })
        else:
            del MOBILE_VERIFICATION_TOKENS[token] # Clean up if email fails
            return jsonify({"success": False, "error": "Failed to send email"}), 500
    except Exception as e:
        app.logger.error(f"Error: {str(e)}")
        return jsonify({"success": False, "error": "Internal error"}), 500

@app.route("/mobile-verify/<token>")
def mobile_verify(token):
    try:
        if token not in MOBILE_VERIFICATION_TOKENS:
            return render_template("mobile_verify.html", error="Invalid token")
        verification_data = MOBILE_VERIFICATION_TOKENS[token]
        if verification_data["status"] != "pending":
            return render_template("mobile_verify.html", error="Already completed")

        return render_template("mobile_verify.html", token=token, email=verification_data["email"])
    except Exception as e:
        app.logger.error(f"Error: {str(e)}")
        return render_template("mobile_verify.html", error="An error occurred")

@app.route("/api/verify-face", methods=["POST"])
def verify_face():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "No data"}), 400

        token = data.get("token")
        captured_image = data.get("image")
        if not token or not captured_image:
            return jsonify({"success": False, "error": "Token and image required"}), 400

        if token not in MOBILE_VERIFICATION_TOKENS:
            return jsonify({"success": False, "error": "Invalid token"}), 400

        verification_data = MOBILE_VERIFICATION_TOKENS[token]
        if verification_data["status"] != "pending":
            return jsonify({"success": False, "error": "Already completed"}), 400

        img_bgr, err = _decode_base64_image(captured_image)
        if img_bgr is None:
            return jsonify({"success": False, "error": f"Decode failed: {err}"}), 400

        user = fetchone("SELECT face_encoding FROM users WHERE id=%s", (verification_data["user_id"],))
        if not user or not user.get("face_encoding"):
            return jsonify({"success": False, "error": "No face registered"}), 400

        live_enc_bytes, _, err = _face_encoding_from_bgr_image(img_bgr)
        if live_enc_bytes is None:
            return jsonify({"success": False, "error": f"Face detection: {err}"}), 400

        dist = enc_distance(live_enc_bytes, user["face_encoding"])
        if dist > 0.55: # Threshold for mismatch
            verification_data["status"] = "failed"
            verification_data["distance"] = dist
            return jsonify({"success": False, "error": "Face mismatch", "distance": dist}), 401

        verification_data["status"] = "success"
        verification_data["distance"] = dist
        return jsonify({"success": True, "message": "Verified", "distance": dist})
    except Exception as e:
        app.logger.error(f"Error: {str(e)}")
        return jsonify({"success": False, "error": "Internal error"}), 500

@app.route("/api/check-verification-status/<token>")
def check_verification_status(token):
    try:
        if token not in MOBILE_VERIFICATION_TOKENS:
            return jsonify({"status": "expired"})

        status = MOBILE_VERIFICATION_TOKENS[token]["status"]
        if status == "success":
            user_id = MOBILE_VERIFICATION_TOKENS[token]["user_id"]
            session["user_id"] = user_id
            session.modified = True
            del MOBILE_VERIFICATION_TOKENS[token] # Clean up after success
            return jsonify({"status": "success", "redirect_url": url_for("index")})
        elif status == "failed":
            del MOBILE_VERIFICATION_TOKENS[token] # Clean up after failure
            return jsonify({"status": "failed"})
        else:
            return jsonify({"status": "pending"})
    except Exception as e:
        app.logger.error(f"Error: {str(e)}")
        return jsonify({"status": "error"})

@app.route("/")
def index():
    featured = fetchall(
        "SELECT p.*, "
        "CASE WHEN p.image_blob IS NOT NULL THEN CONCAT('/product-image/', p.id) ELSE p.image_url END AS image_url, "
        "c.name AS category_name "
        "FROM products p JOIN categories c ON c.id = p.category_id "
        "ORDER BY p.created_at DESC LIMIT 8"
    )
    categories = fetchall("SELECT * FROM categories ORDER BY name")
    return render_template("index.html", featured=featured, categories=categories)

@app.route("/categories")
def categories():
    cats = fetchall("SELECT * FROM categories ORDER BY name")
    return render_template("categories.html", categories=cats)

@app.route("/category/<slug>")
def category(slug):
    cat = fetchone("SELECT * FROM categories WHERE slug=%s", (slug,))
    if not cat:
        abort(404)
    prods = fetchall(
        "SELECT p.*, "
        "CASE WHEN p.image_blob IS NOT NULL THEN CONCAT('/product-image/', p.id) ELSE p.image_url END AS image_url, "
        "c.name AS category_name "
        "FROM products p "
        "JOIN categories c ON c.id = p.category_id "
        "WHERE c.slug = %s ORDER BY p.created_at DESC",
        (slug,),
    )
    return render_template("category.html", category=cat, products=prods)

@app.route("/electronics")
def electronics():
    return category("electronics")

@app.route("/home-appliances")
def home_appliances():
    return category("home-appliances")

@app.route("/search")
def search():
    q = request.args.get("q", "").strip()
    results = []
    if q:
        like = f"%{q}%"
        results = fetchall(
            "SELECT p.*, "
            "CASE WHEN p.image_blob IS NOT NULL THEN CONCAT('/product-image/', p.id) ELSE p.image_url END AS image_url, "
            "c.name AS category_name "
            "FROM products p "
            "JOIN categories c ON c.id = p.category_id "
            "WHERE p.name LIKE %s OR p.description LIKE %s "
            "ORDER BY p.created_at DESC",
            (like, like),
        )
    return render_template("search.html", q=q, results=results)

# CART / WISHLIST routes (unchanged)...
@app.route("/add-to-cart/<int:product_id>", methods=["POST"])
def add_to_cart(product_id):
    qty = max(1, int(request.form.get("qty", 1)))
    cart_init()

    # Handle old format conversion if needed
    if isinstance(session["cart"].get(str(product_id)), int):
        current_qty = session["cart"][str(product_id)]
        session["cart"][str(product_id)] = {"qty": current_qty, "warranty_ext": 0}

    item_data = session["cart"].get(str(product_id), {"qty": 0, "warranty_ext": 0})
    item_data["qty"] = item_data.get("qty", 0) + qty
    session["cart"][str(product_id)] = item_data
    session.modified = True

    flash("Added to cart.", "success")

    # Return cart_count for AJAX
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return jsonify({
            "success": True,
            "cart_count": get_cart_count(),
            "message": "Added to cart."
        })
    return redirect(request.referrer or url_for("index"))

@app.route("/remove_from_cart", methods=["POST"])
def remove_from_cart():
    if request.is_json:
        data = request.get_json()
        product_id = str(data["product_id"])
    else:
        product_id = str(request.form.get("product_id"))

    cart = session.get("cart", {})
    cart.pop(product_id, None)
    session["cart"] = cart
    session.modified = True

    items, subtotal = cart_items()
    return jsonify({
        "success": True,
        "subtotal": float(subtotal),
        "cart_count": get_cart_count()
    })

@app.route("/cart")
def cart():
    items, subtotal = cart_items()
    return render_template("cart.html", items=items, subtotal=subtotal)

@app.route("/update-cart", methods=["POST"])
def update_cart():
    cart_init()
    if request.is_json:
        data = request.get_json()
        for item in data.get("items", []):
            pid = str(item.get("product_id"))
            qty = max(0, int(item.get("qty", 0)))

            # Handle old format conversion if needed
            if isinstance(session["cart"].get(pid), int):
                current_qty = session["cart"][pid]
                session["cart"][pid] = {"qty": current_qty, "warranty_ext": 0}
            item_data = session["cart"].get(pid, {"qty": 0, "warranty_ext": 0})

            if qty == 0:
                session["cart"].pop(pid, None)
            else:
                item_data["qty"] = qty
                session["cart"][pid] = item_data
    else:
        for key, value in request.form.items():
            if key.startswith("qty_"):
                pid = key.split("_", 1)[1]
                qty = max(0, int(value or 0))

                # Handle old format conversion if needed
                if isinstance(session["cart"].get(pid), int):
                    current_qty = session["cart"][pid]
                    session["cart"][pid] = {"qty": current_qty, "warranty_ext": 0}
                item_data = session["cart"].get(pid, {"qty": 0, "warranty_ext": 0})

                if qty == 0:
                    session["cart"].pop(pid, None)
                else:
                    item_data["qty"] = qty
                    session["cart"][pid] = item_data
            elif key.startswith("warranty_extension_"):
                pid = key.split("_", 2)[2] # Split "warranty_extension_<id>"
                ext_years = max(0, int(value or 0))

                # Handle old format conversion if needed
                if isinstance(session["cart"].get(pid), int):
                    current_qty = session["cart"][pid]
                    session["cart"][pid] = {"qty": current_qty, "warranty_ext": 0}
                item_data = session["cart"].get(pid, {"qty": 0, "warranty_ext": 0})
                item_data["warranty_ext"] = ext_years
                session["cart"][pid] = item_data
                print(f"Updated warranty extension for {pid} to {ext_years}") # Debug log

    session.modified = True
    items, subtotal = cart_items()
    return jsonify({
        "success": True,
        "subtotal": float(subtotal),
        "cart_count": get_cart_count()
    })

# WISHLIST routes unchanged...
@app.route("/wishlist")
@login_required
def wishlist():
    wishlist_items = []
    user_id = session["user_id"]
    items = fetchall("""
        SELECT w.id as wishlist_id, p.*, 
        CASE WHEN p.image_blob IS NOT NULL THEN CONCAT('/product-image/', p.id) ELSE p.image_url END AS image_url
        FROM wishlist w
        JOIN products p ON p.id = w.product_id
        WHERE w.user_id = %s
        ORDER BY w.created_at DESC
    """, (user_id,))
    return render_template("wishlist.html", wishlist_items=items)

@app.route("/add-to-wishlist/<int:product_id>", methods=["POST"])
@login_required
def add_to_wishlist(product_id):
    user_id = session["user_id"]
    existing = fetchone(
        "SELECT id FROM wishlist WHERE user_id = %s AND product_id = %s", 
        (user_id, product_id)
    )
    if not existing:
        execute(
            "INSERT INTO wishlist (user_id, product_id) VALUES (%s, %s)",
            (user_id, product_id)
        )
        flash("Added to wishlist!", "success")
    else:
        flash("Already in your wishlist", "info")

    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return jsonify({
            "success": True,
            "wishlist_count": get_wishlist_count()  # ✅
        })
    return redirect(request.referrer or url_for("index"))

@app.route("/remove-from-wishlist/<int:wishlist_id>", methods=["POST"])
@login_required
def remove_from_wishlist(wishlist_id):
    user_id = session["user_id"]
    item = fetchone(
        "SELECT id FROM wishlist WHERE id = %s AND user_id = %s", 
        (wishlist_id, user_id)
    )
    if item:
        execute("DELETE FROM wishlist WHERE id = %s", (wishlist_id,))
        flash("Removed from wishlist", "success")
    else:
        flash("Item not found", "error")

    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return jsonify({
            "success": True,
            "wishlist_count": get_wishlist_count()  # ✅
        })
    return redirect(url_for("wishlist"))

@app.route("/move-wishlist-to-cart/<int:wishlist_id>", methods=["POST"])
@login_required
def move_wishlist_to_cart(wishlist_id):
    user_id = session["user_id"]
    item = fetchone("""
        SELECT w.product_id, p.name 
        FROM wishlist w
        JOIN products p ON p.id = w.product_id
        WHERE w.id = %s AND w.user_id = %s
    """, (wishlist_id, user_id))
    if not item:
        return jsonify({"success": False, "error": "Item not found"})

    cart_init()
    product_id = str(item["product_id"])
    session["cart"][product_id] = session["cart"].get(product_id, 0) + 1
    session.modified = True

    execute("DELETE FROM wishlist WHERE id = %s", (wishlist_id,))

    return jsonify({
        "success": True,
        "message": f"{item['name']} added to cart!",
        "cart_count": get_cart_count(),
        "wishlist_count": get_wishlist_count()  # ✅ Sync both
    })

# ====== CHECKOUT ROUTE (MODIFIED: create payment session, not order) ======
@app.route("/checkout", methods=["GET", "POST"])
@login_required
def checkout():
    items, subtotal = cart_items() # This now fetches warranty extensions too
    if not items:
        flash("Cart is empty.", "warning")
        return redirect(url_for("cart"))

    user = fetchone("""
        SELECT name, email, mobile, address_line1, address_line2, city, state, pincode 
        FROM users WHERE id = %s
    """, (session["user_id"],))

    if request.method == "POST":
        # Face verification (same as before)
        flash("Confirming identity via face verification.", "info")
        row = fetchone("SELECT face_encoding FROM users WHERE id=%s", (session["user_id"],))
        if not row or row["face_encoding"] is None:
            flash("No face registered.", "danger")
            return render_template("checkout.html", items=items, subtotal=subtotal, user=user)

        live_enc, face_img, err = capture_face_encoding_server(window_title="Checkout: verify identity")
        if live_enc is None:
            flash(f"Face verification failed: {err}", "danger")
            return render_template("checkout.html", items=items, subtotal=subtotal, user=user)

        dist = enc_distance(live_enc, row["face_encoding"])
        if dist > 0.55:
            flash("Face mismatch. Checkout blocked.", "danger")
            return render_template("checkout.html", items=items, subtotal=subtotal, user=user)

        # Collect shipping details
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        address = request.form.get("address", "").strip()
        city = request.form.get("city", "").strip()
        state = request.form.get("state", "").strip()
        pincode = request.form.get("pincode", "").strip()

        if not all([name, email, address, city, state, pincode]):
            flash("Fill all fields.", "danger")
            return render_template("checkout.html", items=items, subtotal=subtotal, user=user)

        # Create a mock payment session instead of inserting order immediately
        token = secrets.token_urlsafe(24)
        # Snapshot items (serialize only what's necessary)
        snapshot_items = []
        for it in items:
            snapshot_items.append({
                "product": {
                    "id": it["product"]["id"],
                    "name": it["product"]["name"],
                    "price": str(it["product"]["price"]),
                    "warranty_years": it["product"].get("warranty_years", 0)
                },
                "qty": int(it["qty"]),
                "warranty_extension": int(it.get("warranty_extension", 0)),
                "line_total": str(it.get("line_total", "0"))
            })

        MOCK_PAYMENT_SESSIONS[token] = {
            "user_id": session.get("user_id"),
            "created": time.time(),
            "expires": time.time() + MOCK_PAYMENT_TTL,
            "customer": {
                "name": name,
                "email": email,
                "address": address,
                "city": city,
                "state": state,
                "pincode": pincode
            },
            "items": snapshot_items,
            "subtotal": str(subtotal)
        }

        # Redirect to mock gateway page for payment (GET)
        return redirect(url_for("mock_gateway_token", token=token))

    return render_template("checkout.html", items=items, subtotal=subtotal, user=user)

# ====== Mock Gateway Routes ======
@app.route("/mock-gateway/<token>", methods=["GET"])
@login_required
def mock_gateway_token(token):
    data = MOCK_PAYMENT_SESSIONS.get(token)
    if not data or data.get("expires", 0) < time.time():
        MOCK_PAYMENT_SESSIONS.pop(token, None)
        flash("Payment session expired or invalid.", "danger")
        return redirect(url_for("cart"))

    # DEBUG: Print what's in the data
    print(f"\n=== DEBUG: Payment data for token {token} ===")
    print(f"Type of data: {type(data)}")
    print(f"Data keys: {list(data.keys())}")
    print(f"Has 'items' key: {'items' in data}")
    if 'items' in data:
        print(f"Type of items: {type(data['items'])}")
        if isinstance(data['items'], list):
            print(f"Number of items: {len(data['items'])}")
            for i, item in enumerate(data['items']):
                print(f"  Item {i}: {item}")
        else:
            print(f"Items value: {data['items']}")
    print("=== END DEBUG ===\n")
    
    # Also print the customer info
    print(f"Customer data: {data.get('customer')}")

    return render_template("mock_gateway.html", token=token, payment=data)

@app.route("/mock-gateway", methods=["GET"])  # ← ADD THIS NEW ROUTE
def mock_gateway_redirect():
    """Redirect old /mock-gateway links to checkout"""
    flash("Please proceed to checkout first.", "info")
    return redirect(url_for("checkout"))

@app.route("/mock-gateway/complete/<token>", methods=["POST"])
@login_required
def mock_gateway_complete(token):
    data = MOCK_PAYMENT_SESSIONS.get(token)
    if not data or data.get("expires", 0) < time.time():
        MOCK_PAYMENT_SESSIONS.pop(token, None)
        flash("Payment session expired or invalid.", "danger")
        return redirect(url_for("cart"))

    try:
        user_id = data["user_id"]
        cust = data["customer"]
        items = data["items"]
        subtotal = Decimal(str(data["subtotal"]))

        # Create the order now that "payment succeeded"
        order_id = execute(
            "INSERT INTO orders (user_id, customer_name, customer_email, address_line, city, state, pincode, total_amount, created_at) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,NOW())",
            (user_id, cust["name"], cust["email"], cust["address"], cust["city"], cust["state"], cust["pincode"], str(subtotal))
        )

        # Insert items and update stock
        for it in items:
            p = it["product"]
            qty = int(it.get("qty", 1))
            base_warranty = int(p.get("warranty_years", 0) or 0)
            extension_years = int(it.get("warranty_extension", 0) or 0)
            effective_warranty = base_warranty + extension_years

            execute(
                "INSERT INTO order_items (order_id, product_id, quantity, unit_price, effective_warranty_years) VALUES (%s,%s,%s,%s,%s)",
                (order_id, p["id"], qty, str(p["price"]), effective_warranty)
            )
            # update stock safely
            execute("UPDATE products SET stock = GREATEST(stock - %s, 0) WHERE id=%s", (qty, p["id"]))

        # Clear cart after successful order creation
        session["cart"] = {}
        session.modified = True

        # Clean up mock session
        MOCK_PAYMENT_SESSIONS.pop(token, None)

        # Try sending email (best-effort)
        try:
            send_email(cust["email"], f"Order #{order_id} Confirmed", "order_confirmation",
                       user_name=cust["name"], order_id=order_id, order_items=items, total_amount=subtotal)
            flash("Order placed! Confirmation email sent.", "success")
        except Exception:
            flash("Order placed! Email not sent (not configured).", "warning")

        return redirect(url_for("order_success", order_id=order_id))
    except Exception as e:
        app.logger.exception(f"Mock payment completion failed: {e}")
        flash("An error occurred while placing your order. Check logs.", "danger")
        return redirect(url_for("cart"))
    
# Order success (unchanged)
@app.route("/order-success/<int:order_id>")
@login_required
def order_success(order_id):
    order = fetchone("SELECT * FROM orders WHERE id=%s", (order_id,))
    items = fetchall(
        "SELECT oi.*, p.name FROM order_items oi JOIN products p ON p.id = oi.product_id WHERE oi.order_id=%s",
        (order_id,),
    )
    return render_template("order_success.html", order=order, items=items)

# ====== PROFILE ROUTES (ADDED) ======
@app.route("/profile")
@login_required
def profile():
    user = current_user()
    if not user:
        flash("Please login first.", "warning")
        return redirect(url_for("login"))
    
    # Get user's complete details
    user_details = fetchone("""
        SELECT u.*, 
               CASE WHEN u.face_image IS NOT NULL THEN u.face_image 
               ELSE '/static/default-avatar.jpg' END as face_image 
        FROM users u WHERE id=%s
    """, (user["id"],))
    
    # Get user's orders with product details
    orders = fetchall("""
        SELECT o.id, o.total_amount as total, o.created_at as date_time,
               oi.product_id, oi.quantity, oi.unit_price, oi.effective_warranty_years as warranty_years,
               p.name
        FROM orders o
        JOIN order_items oi ON o.id = oi.order_id
        JOIN products p ON p.id = oi.product_id
        WHERE o.user_id = %s
        ORDER BY o.created_at DESC
    """, (user["id"],))
    
    # Group orders by order id for the template
    grouped_orders = {}
    for order in orders:
        order_id = order["id"]
        if order_id not in grouped_orders:
            grouped_orders[order_id] = {
                "id": order_id,
                "total": order["total"],
                "date_time": order["date_time"],
                "products": []
            }
        grouped_orders[order_id]["products"].append({
            "product_id": order["product_id"],
            "name": order["name"],
            "quantity": order["quantity"],
            "unit_price": order["unit_price"],
            "warranty_years": order["warranty_years"]
        })
    
    # Convert to list
    orders_list = list(grouped_orders.values())
    
    # Get warranty claims (if table exists)
    claims = []
    try:
        claims = fetchall("""
            SELECT wc.*, p.name as product_name
            FROM warranty_claims wc
            JOIN products p ON p.id = wc.product_id
            WHERE wc.user_id = %s
            ORDER BY wc.created_at DESC
        """, (user["id"],))
    except Exception as e:
        app.logger.warning(f"Warranty claims table might not exist: {e}")
        claims = []
    
    # Get shop info for the invoice modal
    shop_info = {}
    try:
        shop_info = fetchone("SELECT shop_name, shop_logo, shop_tagline, shop_contact FROM site_settings LIMIT 1")
    except Exception as e:
        app.logger.warning(f"Site settings table might not exist: {e}")
        shop_info = {}
    
    return render_template("profile.html", 
                         user=user_details, 
                         orders=orders_list, 
                         claims=claims,
                         shop_name=shop_info.get("shop_name", "Shop Ease"),
                         shop_logo=shop_info.get("shop_logo", None),
                         shop_tagline=shop_info.get("shop_tagline", None),
                         shop_contact=shop_info.get("shop_contact", None))

@app.route("/profile/face-request-otp")
@login_required
def profile_face_request_otp():
    user = current_user()
    if not user:
        flash("Please login first.", "warning")
        return redirect(url_for("login"))
    
    # Generate a token for face update
    token = secrets.token_urlsafe(24)
    FACE_UPDATE_TOKENS[token] = {
        "user_id": user["id"],
        "timestamp": time.time(),
        "email": user["email"],
        "name": user["name"]
    }
    
    # Send email with verification link
    sent = send_email(
        to=user["email"],
        subject="Face Update Verification",
        template="face_update_verification",
        user_name=user["name"],
        verification_url=url_for('update_face_verify', token=token, _external=True)
    )
    
    if sent:
        flash("A verification link has been sent to your email. Please check your inbox.", "info")
    else:
        flash("Failed to send verification email. Please try again.", "danger")
    
    return redirect(url_for("profile"))

@app.route("/update-face/<token>")
@login_required
def update_face_verify(token):
    if token not in FACE_UPDATE_TOKENS:
        flash("Invalid or expired verification link.", "danger")
        return redirect(url_for("profile"))
    
    data = FACE_UPDATE_TOKENS[token]
    if time.time() - data["timestamp"] > FACE_UPDATE_CODE_TTL:
        del FACE_UPDATE_TOKENS[token]
        flash("Verification link expired.", "warning")
        return redirect(url_for("profile"))
    
    # Store token in session for the update process
    session["face_update_token"] = token
    session.modified = True
    
    return render_template("update_face.html", token=token)

@app.route("/update-face-complete", methods=["POST"])
@login_required
def update_face_complete():
    token = session.get("face_update_token")
    if not token or token not in FACE_UPDATE_TOKENS:
        flash("Invalid session. Please request a new verification.", "danger")
        return redirect(url_for("profile"))
    
    data = FACE_UPDATE_TOKENS[token]
    user_id = data["user_id"]
    
    # Capture new face
    flash("Please align your face for capture.", "info")
    enc_bytes, cropped_bgr, err = capture_face_encoding_server(window_title="Update Face: Capture New Face")
    
    if enc_bytes is None:
        flash(f"Face capture failed: {err}", "danger")
        return redirect(url_for("update_face_verify", token=token))
    
    # Update database
    execute("UPDATE users SET face_encoding=%s WHERE id=%s", (enc_bytes, user_id))
    
    if cropped_bgr is not None:
        rel_path = _save_face_image_bytes(user_id, cropped_bgr)
        if rel_path:
            execute("UPDATE users SET face_image=%s WHERE id=%s", (rel_path, user_id))
    
    # Clean up
    del FACE_UPDATE_TOKENS[token]
    session.pop("face_update_token", None)
    session.modified = True
    
    flash("Face updated successfully!", "success")
    return redirect(url_for("profile"))


@app.route("/edit-profile", methods=["GET", "POST"])
@login_required
def edit_profile():
    user = current_user()  # Note: Added parentheses to CALL the function
    
    if not user:
        flash("Please login first.", "warning")
        return redirect(url_for("login"))
    
    # Get complete user details
    user_details = fetchone("SELECT * FROM users WHERE id=%s", (user["id"],))
    
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        mobile = request.form.get("mobile", "").strip()
        address_line1 = request.form.get("address_line1", "").strip()
        address_line2 = request.form.get("address_line2", "").strip()
        city = request.form.get("city", "").strip()
        state = request.form.get("state", "").strip()
        pincode = request.form.get("pincode", "").strip()
        
        # Validate required fields
        if not name or not email:
            flash("Name and email are required.", "danger")
            return render_template("edit_profile.html", user=user_details)
        
        # Check if email is being changed and if it's already taken by another user
        if email != user_details["email"]:
            existing = fetchone("SELECT id FROM users WHERE email=%s AND id!=%s", (email, user["id"]))
            if existing:
                flash("Email already in use by another account.", "danger")
                return render_template("edit_profile.html", user=user_details)
        
        # Validate mobile number if provided
        if mobile and (not mobile.isdigit() or len(mobile) not in (10, 11, 12, 13)):
            flash("Please enter a valid mobile number (10-13 digits).", "danger")
            return render_template("edit_profile.html", user=user_details)
        
        try:
            # Update the user's profile
            execute("""
                UPDATE users 
                SET name=%s, email=%s, mobile=%s, 
                    address_line1=%s, address_line2=%s, 
                    city=%s, state=%s, pincode=%s 
                WHERE id=%s
            """, (name, email, mobile, address_line1, address_line2, 
                  city, state, pincode, user["id"]))
            
            flash("Profile updated successfully!", "success")
            
            # If email was changed, update session if needed
            if email != user_details["email"]:
                # You might want to force re-login or send verification email
                send_email(
                    to=email,
                    subject="Profile Email Updated",
                    template="email_changed",
                    user_name=name,
                    old_email=user_details["email"],
                    new_email=email
                )
                flash("Email updated. A notification has been sent to your new email.", "info")
            
            # Redirect to profile page
            return redirect(url_for("profile"))
            
        except Exception as e:
            app.logger.error(f"Failed to update profile: {e}")
            flash("Failed to update profile. Please try again.", "danger")
            return render_template("edit_profile.html", user=user_details)
    
    # GET request - show the edit form
    return render_template("edit_profile.html", user=user_details)

# ====== REGISTRATION & LOGIN ROUTES ======
def _gen_code(length=6):
    chars = string.ascii_uppercase + string.digits + string.ascii_lowercase
    return ''.join(random.choice(chars) for _ in range(length))

@app.route("/register", methods=["GET", "POST"])
def register():
    pre_name = ""
    pre_email = ""
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        pre_name = name
        pre_email = email

        if not name or not email:
            flash("Name and email are required.", "danger")
            return render_template("auth_register.html", name=pre_name, email=pre_email)

        exists = fetchone("SELECT id FROM users WHERE email=%s", (email,))
        if exists:
            flash("Email already registered.", "warning")
            return render_template("auth_register.html", name=pre_name, email=pre_email)

        if not password:
            password = secrets.token_urlsafe(12) # Generate secure temp password

        token = secrets.token_urlsafe(24)
        code = _gen_code(6)
        EMAIL_VERIFICATION_TOKENS[token] = {
            "name": name,
            "email": email,
            "password_hash": generate_password_hash(password),
            "code": code,
            "timestamp": time.time(),
            "attempts": 0,
        }

        sent = send_email(
            to=email,
            subject="Your verification code",
            template="email_verification_code",
            user_name=name,
            code=code,
            ttl_minutes=int(VERIFICATION_CODE_TTL/60),
        )
        if sent:
            flash("A 6-character verification code has been sent to your email. Enter it below.", "info")
            return render_template("enter_verification_code.html", token=token, email=email)
        else:
            EMAIL_VERIFICATION_TOKENS.pop(token, None) # Clean up if email fails
            flash("Failed to send verification code. Please try again later.", "danger")
            return render_template("auth_register.html", name=pre_name, email=pre_email)

    return render_template("auth_register.html", name=pre_name, email=pre_email)

@app.route("/verify-code/<token>", methods=["GET", "POST"])
def verify_code(token):
    data = EMAIL_VERIFICATION_TOKENS.get(token)
    if not data:
        flash("Invalid or expired verification session. Please register again.", "danger")
        return redirect(url_for("register"))

    if time.time() - data["timestamp"] > VERIFICATION_CODE_TTL:
        EMAIL_VERIFICATION_TOKENS.pop(token, None)
        flash("Verification code expired. Please register again.", "warning")
        return redirect(url_for("register"))

    if request.method == "POST":
        entered = request.form.get("code", "").strip()
        data["attempts"] = data.get("attempts", 0) + 1

        if data["attempts"] > VERIFICATION_MAX_ATTEMPTS:
            EMAIL_VERIFICATION_TOKENS.pop(token, None)
            flash("Too many failed attempts. Please register again.", "danger")
            return redirect(url_for("register"))

        if entered == data.get("code"):
            try:
                uid = execute(
                    "INSERT INTO users (name, email, password_hash, email_verified, profile_complete) VALUES (%s,%s,%s,1,0)",
                    (data["name"], data["email"], data["password_hash"])
                )
                session["user_id"] = uid  # <-- FIX: Log user in immediately
                session.modified = True
                EMAIL_VERIFICATION_TOKENS.pop(token, None) # Clean up after success
                flash("Email verified! Please complete your profile.", "success")
                return redirect(url_for("complete_profile"))
            except Exception as e:
                app.logger.error(f"User creation failed after code verification: {e}")
                flash("An error occurred creating your account. Please try again.", "danger")
                return redirect(url_for("register"))
        else:
            EMAIL_VERIFICATION_TOKENS[token] = data # Update attempts count
            attempts_left = VERIFICATION_MAX_ATTEMPTS - data["attempts"]
            flash(f"Incorrect code. Attempts left: {attempts_left}", "danger")
            return render_template("enter_verification_code.html", token=token, email=data["email"])

    return render_template("enter_verification_code.html", token=token, email=data["email"])

@app.route("/api/verify-code/<token>", methods=["POST"])
def api_verify_code(token):
    data = EMAIL_VERIFICATION_TOKENS.get(token)
    if not data:
        return jsonify({"success": False, "error": "Invalid or expired token"}), 400

    if time.time() - data["timestamp"] > VERIFICATION_CODE_TTL:
        EMAIL_VERIFICATION_TOKENS.pop(token, None)
        return jsonify({"success": False, "error": "Expired token"}), 400

    entered = request.form.get("code") or (request.get_json() or {}).get("code")
    if not entered:
        return jsonify({"success": False, "error": "Code required"}), 400

    data["attempts"] = data.get("attempts", 0) + 1
    if data["attempts"] > VERIFICATION_MAX_ATTEMPTS:
        EMAIL_VERIFICATION_TOKENS.pop(token, None)
        return jsonify({"success": False, "error": "Too many attempts"}), 400

    if entered == data["code"]:
        try:
            uid = execute(
                "INSERT INTO users (name, email, password_hash, email_verified, profile_complete) VALUES (%s,%s,%s,1,0)",
                (data["name"], data["email"], data["password_hash"])
            )
            EMAIL_VERIFICATION_TOKENS.pop(token, None) # Clean up after success
            session["user_id"] = uid  # <-- FIX: Log user in immediately
            session.modified = True
            return jsonify({"success": True, "redirect": url_for("complete_profile")})
        except Exception as e:
            app.logger.error(f"API user creation failed: {e}")
            return jsonify({"success": False, "error": "DB error"}), 500
    else:
        EMAIL_VERIFICATION_TOKENS[token] = data # Update attempts count
        return jsonify({"success": False, "error": "Incorrect code", "attempts_left": VERIFICATION_MAX_ATTEMPTS - data["attempts"]}), 401

@app.route("/complete-profile", methods=["GET", "POST"])
@login_required
def complete_profile():
    user_id = session.get("user_id")
    if not user_id:
        flash("Please login first.", "warning")
        return redirect(url_for("login"))
    
    # Check if profile is already complete
    user = fetchone("SELECT profile_complete FROM users WHERE id=%s", (user_id,))
    if not user:
        flash("User not found.", "danger")
        return redirect(url_for("login"))
    
    # If profile is already complete, redirect to profile page
    if user.get("profile_complete"):
        flash("Your profile is already complete!", "info")
        return redirect(url_for("profile"))
    
    if request.method == "POST":
        mobile = request.form.get("mobile", "").strip()
        address_line1 = request.form.get("address_line1", "").strip()
        address_line2 = request.form.get("address_line2", "").strip()
        city = request.form.get("city", "").strip()
        state = request.form.get("state", "").strip()
        pincode = request.form.get("pincode", "").strip()

        if not all([mobile, address_line1, city, state, pincode]):
            flash("All fields marked * are required.", "danger")
            return render_template("complete_profile.html")

        if not mobile.isdigit() or len(mobile) not in (10, 11, 12, 13):
            flash("Please enter a valid mobile number (10-13 digits).", "danger")
            return render_template("complete_profile.html")

        flash("Please allow webcam access to capture your face for login.", "info")
        enc_bytes, cropped_bgr, err = capture_face_encoding_server(window_title="Complete Profile: Capture Face")
        if enc_bytes is None:
            flash(f"Face capture failed: {err}", "danger")
            return render_template("complete_profile.html")

        execute("""
            UPDATE users 
            SET mobile=%s, address_line1=%s, address_line2=%s, city=%s, state=%s, pincode=%s, 
                profile_complete=TRUE, face_encoding=%s
            WHERE id=%s
        """, (mobile, address_line1, address_line2, city, state, pincode, enc_bytes, user_id))

        if cropped_bgr is not None:
            rel_path = _save_face_image_bytes(user_id, cropped_bgr)
            if rel_path:
                execute("UPDATE users SET face_image=%s WHERE id=%s", (rel_path, user_id))

        flash("Profile completed! Welcome to the store!", "success")
        return redirect(url_for("profile"))

    return render_template("complete_profile.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        user = fetchone("SELECT * FROM users WHERE email=%s", (email,))
        if not user or not check_password_hash(user["password_hash"], password):
            flash("Invalid credentials.", "danger")
            return render_template("auth_login.html")

        flash("Verifying your face.", "info")
        row = fetchone("SELECT face_encoding FROM users WHERE id=%s", (user["id"],))
        if not row or row["face_encoding"] is None:
            flash("No face registered.", "danger")
            return render_template("auth_login.html")

        live_enc, face_img, err = capture_face_encoding_server(window_title="Login: verify face")
        if live_enc is None:
            flash(f"Face verification failed: {err}", "danger")
            return render_template("auth_login.html")

        dist = enc_distance(live_enc, row["face_encoding"])
        if dist > 0.55:
            flash("Face mismatch. Login blocked.", "danger")
            return render_template("auth_login.html")

        session["user_id"] = user["id"]
        session.modified = True

        # Check if profile is complete, if not redirect to complete profile
        if not user.get("profile_complete", False):
            flash("Please complete your profile setup.", "info")
            return redirect(url_for("complete_profile"))

        send_email(
            to=user["email"],
            subject="New Login Detected – Your Account",
            template="login_alert",
            user_name=user["name"],
            login_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )

        flash("Welcome!", "success")
        return redirect(request.args.get("next") or url_for("index"))

    return render_template("auth_login.html")

@app.route("/api/verify-face-login", methods=["POST"])
def verify_face_login():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "No data"}), 400

        email = data.get("email")
        image_data = data.get("image")
        if not email or not image_data:
            return jsonify({"success": False, "error": "Email and image required"}), 400

        user = fetchone("SELECT id, password_hash, face_encoding, profile_complete FROM users WHERE email=%s", (email,))
        if not user or not user.get("face_encoding"):
            return jsonify({"success": False, "error": "No face registered"}), 400

        img_bgr, err = _decode_base64_image(image_data)
        if img_bgr is None:
            return jsonify({"success": False, "error": "Invalid image"}), 400

        live_enc_bytes, _, err = _face_encoding_from_bgr_image(img_bgr)
        if live_enc_bytes is None:
            return jsonify({"success": False, "error": "No face detected"}), 400

        dist = enc_distance(live_enc_bytes, user["face_encoding"])
        if dist <= 0.55:
            session["user_id"] = user["id"]
            session.modified = True

            send_email(
                to=user["email"],
                subject="New Login Detected – Your Account",
                template="login_alert",
                user_name=user["name"],
                login_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )

            # Check if profile complete
            if not user.get("profile_complete", False):
                return jsonify({
                    "success": True,
                    "redirect_url": url_for("complete_profile"),
                    "cart_count": get_cart_count(),
                    "wishlist_count": get_wishlist_count()
                })

            return jsonify({
                "success": True,
                "redirect_url": url_for("index"),
                "cart_count": get_cart_count(),
                "wishlist_count": get_wishlist_count()  # ✅
            })
        else:
            return jsonify({"success": False, "error": "Face not recognized"})
    except Exception as e:
        app.logger.error(f"Face login error: {e}")
        return jsonify({"success": False, "error": "Server error"}), 500

@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out.", "info")
    return redirect(url_for("index"))

# ====== WARRANTY CLAIM ROUTES ======
@app.route("/claim-warranty")
@login_required
def claim_warranty():
    order_id = request.args.get("order_id")
    product_id = request.args.get("product_id")
    
    if not order_id or not product_id:
        flash("Invalid request parameters.", "danger")
        return redirect(url_for("profile"))
    
    # Verify the order belongs to the current user
    order = fetchone("""
        SELECT o.*, oi.effective_warranty_years as warranty_years, p.name as product_name
        FROM orders o
        JOIN order_items oi ON o.id = oi.order_id
        JOIN products p ON p.id = oi.product_id
        WHERE o.id = %s AND oi.product_id = %s AND o.user_id = %s
    """, (order_id, product_id, session["user_id"]))
    
    if not order:
        flash("Order not found or access denied.", "danger")
        return redirect(url_for("profile"))
    
    # Check if warranty is still valid
    order_date = order["created_at"]
    warranty_years = order.get("warranty_years", 0)
    
    if warranty_years <= 0:
        flash("This product has no warranty.", "warning")
        return redirect(url_for("profile"))
    
    warranty_end = order_date + timedelta(days=warranty_years * 365)
    if datetime.now() > warranty_end:
        flash("Warranty has expired.", "warning")
        return redirect(url_for("profile"))
    
    return render_template("claim_warranty.html", 
                         order=order, 
                         product_id=product_id,
                         product_name=order["product_name"])

@app.route("/submit-warranty-claim", methods=["POST"])
@login_required
def submit_warranty_claim():
    order_id = request.form.get("order_id")
    product_id = request.form.get("product_id")
    reason = request.form.get("reason", "").strip()
    
    if not order_id or not product_id or not reason:
        flash("All fields are required.", "danger")
        return redirect(url_for("claim_warranty", order_id=order_id, product_id=product_id))
    
    # Verify the order belongs to the current user
    order = fetchone("""
        SELECT o.*, oi.effective_warranty_years as warranty_years
        FROM orders o
        JOIN order_items oi ON o.id = oi.order_id
        WHERE o.id = %s AND oi.product_id = %s AND o.user_id = %s
    """, (order_id, product_id, session["user_id"]))
    
    if not order:
        flash("Order not found or access denied.", "danger")
        return redirect(url_for("profile"))
    
    # Face verification for warranty claim
    flash("Verifying your identity for warranty claim...", "info")
    row = fetchone("SELECT face_encoding FROM users WHERE id=%s", (session["user_id"],))
    if not row or row["face_encoding"] is None:
        flash("No face registered. Cannot process warranty claim.", "danger")
        return redirect(url_for("claim_warranty", order_id=order_id, product_id=product_id))

    live_enc, face_img, err = capture_face_encoding_server(window_title="Warranty Claim: Verify Identity")
    if live_enc is None:
        flash(f"Face verification failed: {err}", "danger")
        return redirect(url_for("claim_warranty", order_id=order_id, product_id=product_id))

    dist = enc_distance(live_enc, row["face_encoding"])
    if dist > 0.55:
        flash("Face mismatch. Warranty claim blocked.", "danger")
        return redirect(url_for("claim_warranty", order_id=order_id, product_id=product_id))
    
    # Insert warranty claim
    try:
        execute("""
            INSERT INTO warranty_claims (user_id, order_id, product_id, reason, claim_date, status)
            VALUES (%s, %s, %s, %s, NOW(), 'pending')
        """, (session["user_id"], order_id, product_id, reason))
        
        flash("Warranty claim submitted successfully! We will review it soon.", "success")
    except Exception as e:
        app.logger.error(f"Failed to submit warranty claim: {e}")
        flash("Failed to submit warranty claim. Please try again.", "danger")
    
    return redirect(url_for("profile"))

# ====== UPDATE PROFILE ROUTE ======
@app.route("/update-profile", methods=["GET", "POST"])
@login_required
def update_profile():
    user = current_user()
    if not user:
        flash("Please login first.", "warning")
        return redirect(url_for("login"))
    
    # Get complete user details
    user_details = fetchone("SELECT * FROM users WHERE id=%s", (user["id"],))
    
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        mobile = request.form.get("mobile", "").strip()
        address_line1 = request.form.get("address_line1", "").strip()
        address_line2 = request.form.get("address_line2", "").strip()
        city = request.form.get("city", "").strip()
        state = request.form.get("state", "").strip()
        pincode = request.form.get("pincode", "").strip()
        
        if not name:
            flash("Name is required.", "danger")
            return redirect(url_for("update_profile"))
        
        if mobile and (not mobile.isdigit() or len(mobile) not in (10, 11, 12, 13)):
            flash("Please enter a valid mobile number (10-13 digits).", "danger")
            return redirect(url_for("update_profile"))
        
        execute("""
            UPDATE users 
            SET name=%s, mobile=%s, address_line1=%s, address_line2=%s, 
                city=%s, state=%s, pincode=%s 
            WHERE id=%s
        """, (name, mobile, address_line1, address_line2, city, state, pincode, user["id"]))
        
        flash("Profile updated successfully!", "success")
        return redirect(url_for("profile"))
    
    return render_template("update_profile.html", user=user_details)

# ====== DEBUG ROUTE (ADD THIS) ======
@app.route("/debug-user")
def debug_user():
    user_id = session.get("user_id")
    output = []
    output.append(f"Session ID: {session.get('user_id')}")
    output.append(f"Session all: {dict(session)}")
    
    if user_id:
        user = fetchone("SELECT id, email, profile_complete, face_encoding IS NOT NULL as has_face FROM users WHERE id=%s", (user_id,))
        if user:
            output.append(f"User ID: {user['id']}")
            output.append(f"Email: {user['email']}")
            output.append(f"Profile Complete: {user['profile_complete']}")
            output.append(f"Has Face: {user['has_face']}")
        else:
            output.append("User not found in DB")
    else:
        output.append("No user in session")
    
    return "<br>".join(output)

if __name__ == "__main__":
    app.run(host='10.152.85.231', port=5000, debug=True, ssl_context='adhoc')