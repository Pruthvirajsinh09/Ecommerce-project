-- ----------------------------------------
-- Database: ecommerce_db
-- ----------------------------------------
DROP DATABASE IF EXISTS ecommerce_db;
CREATE DATABASE IF NOT EXISTS ecommerce_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE ecommerce_db;

-- ----------------------------------------
-- Table: categories
-- ----------------------------------------
CREATE TABLE categories (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) NOT NULL UNIQUE,
    slug VARCHAR(100) NOT NULL UNIQUE,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

    INDEX idx_slug (slug)
) ENGINE=InnoDB;

-- ----------------------------------------
-- Table: users
-- ----------------------------------------
CREATE TABLE users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(150) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    face_encoding MEDIUMBLOB,                -- Pickled face encoding (from face_recognition)
    face_image VARCHAR(255),                 -- Path to saved face thumbnail
    is_admin BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

    INDEX idx_email (email),
    INDEX idx_is_admin (is_admin)
) ENGINE=InnoDB;

-- ----------------------------------------
-- Table: products
-- ----------------------------------------
CREATE TABLE products (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(200) NOT NULL,
    description TEXT,
    price DECIMAL(10, 2) NOT NULL,
    image_url VARCHAR(500),                  -- External image URL
    image_blob MEDIUMBLOB,                   -- Inline image data (e.g., uploaded file)
    image_mimetype VARCHAR(50),              -- MIME type if stored as BLOB
    category_id INT NOT NULL,
    stock INT NOT NULL DEFAULT 100,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

    INDEX idx_category (category_id),
    INDEX idx_name (name(50)),
    INDEX idx_price (price),
    FULLTEXT idx_search (name, description),

    CONSTRAINT fk_product_category
        FOREIGN KEY (category_id)
        REFERENCES categories(id)
        ON DELETE CASCADE
) ENGINE=InnoDB;

-- ----------------------------------------
-- Table: orders
-- ----------------------------------------
CREATE TABLE orders (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    customer_name VARCHAR(150) NOT NULL,
    customer_email VARCHAR(150) NOT NULL,
    address_line TEXT NOT NULL,
    city VARCHAR(100) NOT NULL,
    state VARCHAR(100) NOT NULL,
    pincode VARCHAR(20) NOT NULL,
    total_amount DECIMAL(10, 2) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    INDEX idx_user (user_id),
    INDEX idx_created (created_at),
    INDEX idx_email (customer_email),

    CONSTRAINT fk_order_user
        FOREIGN KEY (user_id)
        REFERENCES users(id)
        ON DELETE CASCADE
) ENGINE=InnoDB;

-- ----------------------------------------
-- Table: order_items
-- ----------------------------------------
CREATE TABLE order_items (
    id INT AUTO_INCREMENT PRIMARY KEY,
    order_id INT NOT NULL,
    product_id INT NOT NULL,
    quantity INT NOT NULL,
    unit_price DECIMAL(10, 2) NOT NULL,

    INDEX idx_order (order_id),
    INDEX idx_product (product_id),

    CONSTRAINT fk_item_order
        FOREIGN KEY (order_id)
        REFERENCES orders(id)
        ON DELETE CASCADE,

    CONSTRAINT fk_item_product
        FOREIGN KEY (product_id)
        REFERENCES products(id)
        ON DELETE RESTRICT
) ENGINE=InnoDB;

-- ----------------------------------------
-- Table: verification_tokens
-- Stores temporary tokens for:
--   - Mobile face verification setup
--   - Cross-device login approval
-- Replaces Redis storage
-- ----------------------------------------
CREATE TABLE verification_tokens (
    id INT AUTO_INCREMENT PRIMARY KEY,
    token VARCHAR(255) NOT NULL UNIQUE,            -- URL-safe token
    type ENUM('mobile_verify', 'login_approval') NOT NULL,
    data JSON NOT NULL,                            -- Stores: user_id, email, status, timestamp, etc.
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    INDEX idx_token (token),
    INDEX idx_type (type),
    INDEX idx_created (created_at)
) ENGINE=InnoDB;

-- ----------------------------------------
-- Insert Sample Data (Optional)
-- ----------------------------------------

-- Admin User (password: "admin123" — change in production!)
INSERT INTO users (name, email, password_hash, is_admin) VALUES (
    'Admin User',
    'admin@example.com',
    '$2b$12$XWy.sPl29vY1jwP96Y2yFOmG12V4911p3J9k0jZ1K1J1J1J1J1J1J.', -- hash of "admin123"
    TRUE
);

-- Categories
INSERT INTO categories (name, slug, description) VALUES
('Electronics', 'electronics', 'Smartphones, laptops, headphones, and more.'),
('Home Appliances', 'home-appliances', 'Refrigerators, washing machines, ACs, etc.'),
('Clothing', 'clothing', 'Men, women, and kids apparel.'),
('Books', 'books', 'Fiction, non-fiction, educational.');

-- Products
INSERT INTO products (name, description, price, image_url, category_id, stock) VALUES
('Smartphone X', 'Latest smartphone with 5G and 108MP camera', 699.99, '/static/img/smartphone.jpg', 1, 50),
('Laptop Pro', 'High-performance laptop for developers and designers', 1299.99, '/static/img/laptop.jpg', 1, 30),
('Wireless Earbuds', 'Noise-cancelling Bluetooth earbuds', 149.99, '/static/img/earbuds.jpg', 1, 100),
('Washing Machine', 'Fully automatic 7kg washing machine', 449.99, '/static/img/washing-machine.jpg', 2, 20),
('Novel: The Lost City', 'Adventure novel set in ancient ruins', 12.99, '/static/img/book.jpg', 4, 75);









ydpzwduotoopdneh