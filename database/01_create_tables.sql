USE CarPriceDB;
GO

IF OBJECT_ID('dbo.cars_raw', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.cars_raw (
        car_id INT IDENTITY(1,1) PRIMARY KEY,

        name NVARCHAR(255) NOT NULL,
        year INT NOT NULL,
        selling_price BIGINT NOT NULL,
        km_driven BIGINT NOT NULL,
        fuel NVARCHAR(50) NOT NULL,
        seller_type NVARCHAR(100) NOT NULL,
        transmission NVARCHAR(50) NOT NULL,
        owner NVARCHAR(100) NOT NULL,

        imported_at DATETIME2 NOT NULL
            DEFAULT SYSUTCDATETIME()
    );
END;
GO

SELECT COLUMN_NAME, DATA_TYPE
FROM INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_NAME = 'cars_raw';