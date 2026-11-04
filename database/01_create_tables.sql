CREATE TABLE IF NOT EXISTS public.cars_raw (
    car_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    year INTEGER NOT NULL,
    selling_price BIGINT NOT NULL,
    km_driven BIGINT NOT NULL,
    fuel VARCHAR(50) NOT NULL,
    seller_type VARCHAR(100) NOT NULL,
    transmission VARCHAR(50) NOT NULL,
    owner VARCHAR(100) NOT NULL,
    imported_at TIMESTAMPTZ DEFAULT NOW()
);