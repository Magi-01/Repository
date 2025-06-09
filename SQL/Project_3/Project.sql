#drop database Aisle_Management;

CREATE DATABASE Aisle_Management;
Use Aisle_Management;

#--------------------------------#
-- For Tables
#--------------------------------#

-- Create SuperMarket table
CREATE TABLE SuperMarket (
    SuperMarketID INT PRIMARY KEY,
    SuperMarketName VARCHAR(255) NOT NULL,
    SuperMarketLocation VARCHAR(255) NOT NULL
);

-- Create SuperMarket table
CREATE TABLE Aisle (
    AisleID INT PRIMARY KEY,
    SuperMarketID INT NOT NULL,
    AisleName VARCHAR(255) NOT NULL,
    FOREIGN KEY (SuperMarketID) REFERENCES SuperMarket(SuperMarketID)
);

-- Create Producer table
CREATE TABLE Producer (
    ProducerID INT PRIMARY KEY,
    ProducerName VARCHAR(255) NOT NULL,
    ProducerLocation VARCHAR(255) NOT NULL
);

-- Create Item table
CREATE TABLE Item (
    ItemID INT PRIMARY KEY,
    ItemName VARCHAR(255) NOT NULL,
    ItemCategory VARCHAR(255),
    ItemStorageType VARCHAR(255),
    ItemPerishable BOOLEAN,
    ItemExpirationDate DATE
);

-- Create Distance table
CREATE TABLE Distance (
    ProducerID INT NOT NULL,
    SuperMarketID INT NOT NULL,
    Distance FLOAT,
    PRIMARY KEY (ProducerID, SuperMarketID),
    FOREIGN KEY (ProducerID) REFERENCES Producer(ProducerID),
    FOREIGN KEY (SuperMarketID) REFERENCES SuperMarket(SuperMarketID)
);

-- Create the Contain Table
CREATE TABLE Contain (
    AisleID INT NOT NULL,
    ItemID INT NOT NULL,
    SuperMarketID INT NOT NULL,
    PRIMARY KEY (AisleID, ItemID),
    FOREIGN KEY (AisleID) REFERENCES Aisle(AisleID),
    FOREIGN KEY (ItemID) REFERENCES Item(ItemID),
    FOREIGN KEY (SuperMarketID) REFERENCES Aisle(SuperMarketID)
);

-- Create the Manufactured_By Table
CREATE TABLE Manufactured_By (
    ItemID INT NOT NULL,
    ProducerID INT NOT NULL,
    PRIMARY KEY (ItemID, ProducerID),
    FOREIGN KEY (ItemID) REFERENCES Item(ItemID),
    FOREIGN KEY (ProducerID) REFERENCES Producer(ProducerID)
);

-- Create ItemLogErrors table
CREATE TABLE ErrorMessages (
    ErrorID INT PRIMARY KEY auto_increment,
    ErrorMessage VARCHAR(255) UNIQUE
);

-- Create ItemLogErrors table
CREATE TABLE ItemLogErrors (
    LogID INT PRIMARY KEY auto_increment,
    ItemID INT Not NULL,
    AisleID INT Not NULL,
    ErrorID INT,
    LogTime DATETIME,
    ToBeThrown BOOLEAN NOT NULL DEFAULT TRUE,
    FOREIGN KEY (ItemID) REFERENCES Item(ItemID),
    FOREIGN KEY (AisleID) REFERENCES Aisle(AisleID),
    FOREIGN KEY (ErrorID) REFERENCES ErrorMessages(ErrorID)
);


#--------------------------------#
-- For Functions and Triggers
#--------------------------------#


DELIMITER $$

CREATE TRIGGER trg_check_item_aisle_count
BEFORE INSERT ON Contain
FOR EACH ROW
BEGIN
    DECLARE aisle_count INT;

    -- Check if the ItemID already exists in another aisle (excluding the same aisle)
    SELECT COUNT(*) INTO aisle_count
    FROM Contain
    WHERE ItemID = NEW.ItemID AND AisleID != NEW.AisleID;

    -- If the item already exists in another aisle, raise an error
    IF aisle_count > 0 THEN
        SIGNAL SQLSTATE '45000'
        SET MESSAGE_TEXT = 'An item can only belong to one aisle.';
    END IF;
END$$

DELIMITER ;


DELIMITER $$

CREATE FUNCTION fn_validate_aisle_compliance(
    item_id INT,
    aisle_id INT,
    supermarket_id INT
)
RETURNS VARCHAR(255)
DETERMINISTIC
BEGIN
    DECLARE item_storage_type VARCHAR(50);
    DECLARE item_category VARCHAR(50);
    DECLARE error_message VARCHAR(255);

    -- Fetch storage type and category
    SELECT ItemStorageType, ItemCategory
    INTO item_storage_type, item_category
    FROM Item
    WHERE ItemID = item_id;

    SET error_message = NULL;

    CASE
        WHEN item_storage_type = 'Frozen' AND
             (
               aisle_id <> (SELECT AisleID FROM Aisle WHERE AisleName = 'Frozen' AND SuperMarketID = supermarket_id LIMIT 1)
               OR
               item_category NOT IN ('Frozen', 'Fresh Meat')
             )
        THEN
            SET error_message = CONCAT(item_category, ' items with Frozen storage must be placed in the Frozen aisle.');

        WHEN item_storage_type = 'Refrigerated' AND
             item_category IN ('Baked', 'Fresh Meat', 'Beverages') AND
             aisle_id <> (SELECT AisleID FROM Aisle WHERE AisleName = 'Refrigerated' AND SuperMarketID = supermarket_id LIMIT 1)
        THEN
            SET error_message = CONCAT(item_category, ' items with Refrigerated storage must be placed in the Refrigerated aisle.');

        WHEN item_storage_type = 'Ambient' AND
             item_category IN ('Refrigerated', 'Frozen')
        THEN
            SET error_message = CONCAT(item_category, ' items cannot have Ambient storage type.');

        WHEN item_category = 'Vegetables' AND
             aisle_id IN (SELECT AisleID FROM Aisle WHERE AisleName IN ('Frozen', 'Refrigerated') AND SuperMarketID = supermarket_id)
        THEN
            SET error_message = 'Vegetables cannot be placed in Frozen or Refrigerated aisles.';

        WHEN item_category = 'Kitchenware' AND
             aisle_id <> (SELECT AisleID FROM Aisle WHERE AisleName = 'Kitchenware' AND SuperMarketID = supermarket_id LIMIT 1)
        THEN
            SET error_message = 'Kitchenware items must be placed in the Kitchenware aisle.';

        WHEN item_category = 'Pet Food' AND
             aisle_id NOT IN (SELECT AisleID FROM Aisle WHERE AisleName IN ('Pet Food', 'Pharmacy') AND SuperMarketID = supermarket_id)
        THEN
            SET error_message = 'Pet Food must be placed in the Pet Food or Pharmacy aisle.';

        WHEN item_category IN ('Baked', 'Beverages') AND
             item_storage_type = 'Frozen'
        THEN
            SET error_message = CONCAT(item_category, ' items cannot be stored in Frozen aisles.');

        WHEN item_category IN ('Dry Food', 'Snacks') AND
             item_storage_type IN ('Frozen', 'Refrigerated')
        THEN
            SET error_message = CONCAT(item_category, ' items cannot be stored in Frozen/Refrigerated aisles.');

        ELSE
            SET error_message = NULL;
    END CASE;

    RETURN error_message;
END$$

DELIMITER ;


DELIMITER $$

CREATE FUNCTION fn_suggest_correct_aisle(item_id INT)
RETURNS INT
DETERMINISTIC
BEGIN
    DECLARE item_storage_type VARCHAR(50);
    DECLARE item_category VARCHAR(50);
    DECLARE suggested_aisle INT;

    -- Fetch storage type and category
    SELECT ItemStorageType, ItemCategory INTO item_storage_type, item_category
    FROM Item
    WHERE ItemID = item_id;

    -- Determine the correct aisle
    CASE
        WHEN item_storage_type = 'Frozen' THEN
            SELECT AisleID INTO suggested_aisle FROM Aisle WHERE AisleName = 'Frozen';
        WHEN item_storage_type = 'Refrigerated' THEN
            SELECT AisleID INTO suggested_aisle FROM Aisle WHERE AisleName = 'Refrigerated';
        WHEN item_category = 'Vegetables' THEN
            SELECT AisleID INTO suggested_aisle FROM Aisle WHERE AisleName = 'Vegetables';
        WHEN item_category = 'Kitchenware' THEN
            SELECT AisleID INTO suggested_aisle FROM Aisle WHERE AisleName = 'Kitchenware';
        WHEN item_category = 'Pet Food' THEN
            SELECT AisleID INTO suggested_aisle FROM Aisle WHERE AisleName IN ('Pet Food', 'Pharmacy') LIMIT 1;
        ELSE
            SELECT AisleID INTO suggested_aisle FROM Aisle WHERE AisleName = 'Ambient';
    END CASE;

    RETURN suggested_aisle;
END$$

DELIMITER ;


DELIMITER $$

CREATE PROCEDURE pr_insert_item_log(item_id INT, aisle_id INT, error_id INT)
BEGIN
    INSERT INTO ItemLogErrors (ItemID, AisleID, LogTime, ErrorID)
    VALUES (item_id, aisle_id, NOW(), error_id);
END$$
DELIMITER ;


DELIMITER $$
CREATE FUNCTION fn_insert_into_error_message(error_message VARCHAR(255))
RETURNS INT
DETERMINISTIC
BEGIN
    DECLARE error_id INT;

    IF NOT EXISTS (
        SELECT 1 FROM ErrorMessages WHERE ErrorMessage = error_message
    ) THEN
        INSERT INTO ErrorMessages (ErrorMessage)
        VALUES (error_message);
    END IF;

    SELECT ErrorID INTO error_id
    FROM ErrorMessages
    WHERE ErrorMessage = error_message;

    RETURN error_id;
END$$

DELIMITER ;


DELIMITER $$

CREATE TRIGGER trg_log_item_wrong_aisle
AFTER INSERT ON Contain
FOR EACH ROW
BEGIN
    DECLARE error_message VARCHAR(255);
    DECLARE error_id INT;

    -- Pass SuperMarketID now
    SET error_message = fn_validate_aisle_compliance(NEW.ItemID, NEW.AisleID, NEW.SuperMarketID);

    IF error_message IS NOT NULL THEN
        SET error_id = fn_insert_into_error_message(error_message);
        CALL pr_insert_item_log(NEW.ItemID, NEW.AisleID, error_id);
    END IF;
END$$

DELIMITER ;

DELIMITER $$

CREATE TRIGGER ReturnItem
AFTER INSERT ON Contain
FOR EACH ROW
BEGIN
    DECLARE maxdistance FLOAT;
    DECLARE checknexpiry DATE;
    DECLARE nexpiry BOOLEAN;
    DECLARE nproducerid INT;
    DECLARE nsupermarketid INT;
    DECLARE nerrorid INT;
    DECLARE ndistance FLOAT;
    DECLARE nperishable BOOLEAN;
    DECLARE error_message VARCHAR(255);

    maxdistance = 7

    SELECT SuperMarketID INTO nsupermarketid FROM Aisle WHERE AisleID = NEW.AisleID LIMIT 1;

    SELECT ProducerID INTO nproducerid FROM manufactured_by WHERE ItemID = NEW.ItemID LIMIT 1;

    SELECT Distance INTO ndistance FROM Distance WHERE SuperMarketID = nsupermarketid AND ProducerID = nproducerid LIMIT 1;

    SELECT ItemPerishable, ItemExpirationDate INTO nperishable, checknexpiry FROM Item WHERE ItemID = NEW.ItemID;

    SET nexpiry = (checknexpiry < NOW());

    IF nexpiry THEN
        IF ndistance <= maxdistance OR nperishable = 0 THEN
            SET error_message = "Expired but returnable due to being non perishable or short distance to producer";
            SET nerrorid = fn_insert_into_error_message(error_message);

            INSERT INTO ItemLogErrors(ItemID, AisleID, ErrorID, LogTime, ToBeThrown)
            VALUES (NEW.ItemID, NEW.AisleID, nerrorid, NOW(), FALSE);

        ELSE
            SET error_message = "Expired and to be thrown away";
            SET nerrorid = fn_insert_into_error_message(error_message);

            INSERT INTO ItemLogErrors(ItemID, AisleID, ErrorID, LogTime, ToBeThrown)
            VALUES (NEW.ItemID, NEW.AisleID, nerrorid, NOW(), TRUE);
        END IF;
    END IF;
END$$

DELIMITER ;


#--------------------------------#
-- For Insertions
#--------------------------------#

INSERT INTO SuperMarket (SuperMarketID, SuperMarketName, SuperMarketLocation)
VALUES 
(1, 'FreshMart', 'Downtown'),
(2, 'GreenGrocer', 'Uptown');

INSERT INTO Aisle (AisleID, SuperMarketID, AisleName)
VALUES 
-- For FreshMart
(1, 1, 'Frozen'),
(2, 1, 'Refrigerated'),
(3, 1, 'Ambient'),
(4, 1, 'Vegetables'),
(5, 1, 'Kitchenware'), -- For GreenGrocer
(6, 2, 'Frozen'),
(7, 2, 'Refrigerated'),
(8, 2, 'Ambient'),
(9, 2, 'Pet Food'),
(10, 2, 'Pharmacy');

INSERT INTO Producer (ProducerID, ProducerName, ProducerLocation)
VALUES 
(1, 'FarmFresh Co.', 'Valley Farm'),
(2, 'CoolDairy Ltd.', 'Mountain Dairy'),
(3, 'SnackWorld Inc.', 'City Snacks Hub');

INSERT INTO Item (ItemID, ItemName, ItemCategory, ItemStorageType, ItemPerishable, ItemExpirationDate)
VALUES 
(1, 'Chicken Breast', 'Fresh Meat', 'Frozen', TRUE, '2025-06-01'),
(2, 'Milk', 'Beverages', 'Refrigerated', TRUE, '2025-05-30'),
(3, 'Frozen Pizza', 'Frozen', 'Frozen', TRUE, '2025-12-31'),
(4, 'Lettuce', 'Vegetables', 'Ambient', TRUE, '2025-05-28'),
(5, 'Dog Food', 'Pet Food', 'Ambient', FALSE, NULL),
(6, 'Canned Beans', 'Dry Food', 'Ambient', FALSE, NULL),
(7, 'Soda', 'Beverages', 'Refrigerated', FALSE, '2026-01-01'),
(8, 'Knife Set', 'Kitchenware', 'Ambient', FALSE, NULL),
(9, 'Ice Cream', 'Frozen', 'Frozen', TRUE, '2025-08-15'),
(10, 'Cookies', 'Snacks', 'Ambient', FALSE, '2026-03-10');

INSERT INTO Contain (AisleID, ItemID, SuperMarketID) VALUES
(1, 1, 1),  -- Chicken Breast in Frozen aisle (FreshMart)
(2, 2, 1),  -- Milk in Refrigerated aisle (FreshMart)
(1, 3, 1),  -- Frozen Pizza in Frozen aisle (FreshMart)
(4, 4, 1),  -- Lettuce in Vegetables aisle (FreshMart)
(9, 5, 2),  -- Dog Food in Pet Food aisle (GreenGrocer)
(8, 6, 2),  -- Canned Beans in Ambient aisle (GreenGrocer)
(7, 7, 2),  -- Soda in Refrigerated aisle (GreenGrocer)
(5, 8, 1),  -- Knife Set in Kitchenware aisle (FreshMart)
(6, 9, 2),  -- Ice Cream in Frozen aisle (GreenGrocer)
(8, 10, 2); -- Cookies in Ambient aisle (GreenGrocer)

INSERT INTO Manufactured_By (ItemID, ProducerID) VALUES
(1, 1),  -- Chicken Breast by FarmFresh Co.
(2, 2),  -- Milk by CoolDairy Ltd.
(3, 3),  -- Frozen Pizza by SnackWorld Inc.
(4, 1),  -- Lettuce by FarmFresh Co.
(5, 3),  -- Dog Food by SnackWorld Inc.
(6, 3),  -- Canned Beans by SnackWorld Inc.
(7, 2),  -- Soda by CoolDairy Ltd.
(8, 3),  -- Knife Set by SnackWorld Inc. (just example)
(9, 3),  -- Ice Cream by SnackWorld Inc.
(10, 3); -- Cookies by SnackWorld Inc.

INSERT INTO Distance (ProducerID, SuperMarketID, Distance) VALUES
(1, 1, 8.5),   -- Happy Farms to FreshMart
(1, 2, 12.3),  -- Happy Farms to GreenGrocer
(2, 1, 5.0),   -- Urban Dairy to FreshMart
(2, 2, 9.7),   -- Urban Dairy to GreenGrocer
(3, 1, 15.2),  -- Global Foods to FreshMart
(3, 2, 6.8);   -- Global Foods to GreenGrocer

#--------------------------------#
-- For Testing Triggers
#--------------------------------#

-- Example of wrong aisle placement
INSERT INTO Contain (AisleID, ItemID) VALUES (6, 4);

#--------------------------------#
-- Joins
#--------------------------------#

CREATE VIEW View_ItemsPerAislePerSuperMarket AS
SELECT
    i.ItemID,
    i.ItemName,
    i.ItemCategory,
    i.ItemStorageType,
    a.AisleID,
    a.AisleName,
    s.SuperMarketID,
    s.SuperMarketName,
    s.SuperMarketLocation
FROM
    Contain c
    JOIN Item i ON c.ItemID = i.ItemID
    JOIN Aisle a ON c.AisleID = a.AisleID
    JOIN SuperMarket s ON a.SuperMarketID = s.SuperMarketID;

SELECT * FROM View_ItemsPerAislePerSuperMarket
WHERE SuperMarketName = 'FreshMart';

SELECT i.ItemID, i.Name AS ItemName, a.Name AS AisleName
FROM Item i
INNER JOIN Contains c ON i.ItemID = c.ItemID
INNER JOIN Aisle a ON c.AisleID = a.AisleID;

-- Left Join
SELECT i.ItemID, i.Name AS ItemName, a.Name AS AisleName
FROM Item i
LEFT JOIN Contains c ON i.ItemID = c.ItemID
LEFT JOIN Aisle a ON c.AisleID = a.AisleID;

-- Right Join
SELECT a.Name AS AisleName, i.ItemID, i.Name AS ItemName
FROM Aisle a
RIGHT JOIN Contains c ON a.AisleID = c.AisleID
RIGHT JOIN Item i ON c.ItemID = i.ItemID;

-- Full/Outer Join
SELECT i.ItemID, i.Name AS ItemName, a.Name AS AisleName
FROM Item i
LEFT JOIN Contains c ON i.ItemID = c.ItemID
LEFT JOIN Aisle a ON c.AisleID = a.AisleID
UNION
SELECT i.ItemID, i.Name AS ItemName, a.Name AS AisleName
FROM Aisle a
RIGHT JOIN Contains c ON a.AisleID = c.AisleID
RIGHT JOIN Item i ON c.ItemID = i.ItemID;

-- Items in Each Aisle
SELECT a.Name AS Aisle, COUNT(c.ItemID) AS NumberOfItems
FROM Aisle a
JOIN Contains c ON a.AisleID = c.AisleID
GROUP BY a.AisleID;

-- Items by Storage Type
SELECT i.StorageType, COUNT(i.ItemID) AS NumberOfItems
FROM Item i
GROUP BY i.StorageType;

-- Items in Each Aisle for Each Storage Type
SELECT a.Name AS Aisle, i.StorageType, COUNT(c.ItemID) AS NumberOfItems
FROM Aisle a
JOIN Contains c ON a.AisleID = c.AisleID
JOIN Item i ON c.ItemID = i.ItemID
GROUP BY a.AisleID, i.StorageType;

-- Items in Each Aisle by Category
SELECT a.Name AS Aisle, i.Category, COUNT(c.ItemID) AS NumberOfItems
FROM Aisle a
JOIN Contains c ON a.AisleID = c.AisleID
JOIN Item i ON c.ItemID = i.ItemID
GROUP BY a.AisleID, i.Category;

-- List of Items in Each Aisle
SELECT i.ItemID, i.Name AS ItemName, a.Name AS AisleName
FROM Item i
JOIN Contains c ON i.ItemID = c.ItemID
JOIN Aisle a ON c.AisleID = a.AisleID;

-- Items That Are Not in the Suggested Aisle
SELECT i.ItemID, i.Name AS ItemName, a.Name AS AisleName,
       CASE
           WHEN i.StorageType = 'Frozen' AND a.Name <> 'Frozen' THEN 'Wrong Aisle'
           WHEN i.StorageType = 'Refrigerated' AND a.Name <> 'Refrigerated' THEN 'Wrong Aisle'
           WHEN i.StorageType = 'Ambient' AND a.Name IN ('Frozen', 'Refrigerated') THEN 'Wrong Aisle'
           ELSE 'Correct Aisle'
       END AS AisleStatus
FROM Item i
JOIN Contains c ON i.ItemID = c.ItemID
JOIN Aisle a ON c.AisleID = a.AisleID;

-- List Items by Storage Type and Aisle
SELECT i.StorageType, a.Name AS AisleName, i.Name AS ItemName
FROM Item i
JOIN Contains c ON i.ItemID = c.ItemID
JOIN Aisle a ON c.AisleID = a.AisleID
ORDER BY i.StorageType, a.Name;



#--------------------------------#
-- Stored Procedures
#--------------------------------#

DELIMITER $$
CREATE PROCEDURE CheckItemCompliance(IN item_id INT, IN aisle_id INT)
BEGIN
    DECLARE item_storage_type VARCHAR(50);
    DECLARE item_category VARCHAR(50);
    DECLARE error_message VARCHAR(255);
    DECLARE suggested_aisle INT;

    -- Fetch storage type and category for the item
    SELECT ItemStorageType, Category INTO item_storage_type, item_category
    FROM Item
    WHERE ItemID = item_id;

    -- Get the suggested aisle from the function
    SET suggested_aisle = fn_suggest_correct_aisle(item_id);

    -- Initialize error message
    SET error_message = NULL;

    -- Check if the aisle provided matches the suggested aisle
    IF aisle_id <> suggested_aisle THEN
        SET error_message = CONCAT('Item with category "', item_category, '" should be placed in aisle ', suggested_aisle, '.');
    END IF;

    -- Output the error message or compliance status
    IF error_message IS NOT NULL THEN
        SELECT error_message AS ComplianceError;
    ELSE
        SELECT 'Item is compliant' AS ComplianceStatus;
    END IF;

    -- The trigger trg_log_item_compliance will automatically handle logging of compliance errors on insert
    -- if a violation is detected and will reject the insert with an error message.
END$$
DELIMITER ;

DELIMITER //
CREATE PROCEDURE sp_check_item_placement()
BEGIN
    DECLARE item_id INT(100);
    DECLARE aisle_id INT(100);
    DECLARE item_storage_type VARCHAR(50);
    DECLARE item_category VARCHAR(50);
    DECLARE aisle_name VARCHAR(50);
    DECLARE suggested_aisle INT;
    DECLARE error_message VARCHAR(255);
    DECLARE error_id INT;

    -- Cursor to iterate over items and their aisle placement
    DECLARE done INT DEFAULT 0;
    DECLARE cur CURSOR FOR
        SELECT I.ItemID, I.StorageType, I.Category, A.Name AS AisleName, C.AisleID
        FROM Item I
        JOIN Contains C ON I.ItemID = C.ItemID
        JOIN Aisle A ON C.AisleID = A.AisleID;

    -- Handler for cursor end
    DECLARE CONTINUE HANDLER FOR NOT FOUND SET done = 1;

    OPEN cur;

    -- Iterate through each row from the cursor
    read_loop: LOOP
        FETCH cur INTO item_id, item_storage_type, item_category, aisle_name, aisle_id;

        IF done THEN
            LEAVE read_loop;
        END IF;

        -- Suggest the correct aisle
        SET suggested_aisle = fn_suggest_correct_aisle(@item_id);

        -- Check if the item is placed in the suggested aisle
        IF (@aisle_id <> suggested_aisle) THEN
            SET error_message = CONCAT(item_category, ' item is not in the suggested aisle: ', suggested_aisle);
        END IF;

        -- If there is an error, log it
        IF error_message IS NOT NULL THEN
            SET error_id = fn_insert_into_error_message(error_message);
            INSERT INTO ItemLogErrors (ItemID, AisleID, LogTime, ErrorID)
            VALUES (@item_id, @aisle_id, NOW(), error_id);
        END IF;
    END LOOP;

    CLOSE cur;
END //
DELIMITER ;

#--------------------------------#
-- To Use Stored Procedures
#--------------------------------#


call CheckItemCompliance(2, 11);
call sp_check_item_placement();


#--------------------------------#
-- For Event Handling
#--------------------------------#


DELIMITER $$
CREATE EVENT ExpirationCheckEvent
ON SCHEDULE EVERY 1 DAY -- Run the event every day
STARTS CURRENT_TIMESTAMP -- Start the event immediately
DO
BEGIN
    DECLARE done INT DEFAULT 0;
    DECLARE item_id INT;
    DECLARE item_name VARCHAR(100);
    DECLARE item_expiration DATETIME;
    DECLARE company_id INT;
    DECLARE company_name VARCHAR(100);
    DECLARE company_email VARCHAR(255);
    DECLARE error_message VARCHAR(255);
    DECLARE error_id INT;

    -- Declare a cursor to fetch expired items
    DECLARE expired_items CURSOR FOR
        SELECT ItemID, Name, ExpirationDate
        FROM Item
        WHERE ExpirationDate <= NOW() AND ExpirationDate > '1000-01-01'; -- Avoid items with invalid expiration date

    -- Declare a handler to exit the cursor loop
    DECLARE CONTINUE HANDLER FOR NOT FOUND SET done = 1;

    OPEN expired_items;

    -- Loop through the expired items
    read_loop: LOOP
        FETCH expired_items INTO item_id, item_name, item_expiration;
        IF done THEN
            LEAVE read_loop;
        END IF;

        -- Get the CompanyID, Name, and Email from the Manufactured_By and Company tables
        SELECT mb.CompanyID, c.Name, c.Email INTO company_id, company_name, company_email
        FROM Manufactured_By mb
        JOIN Company c ON mb.CompanyID = c.CompanyID
        WHERE mb.ItemID = item_id
        LIMIT 1;

        -- Insert log entry if company ID is found
        IF company_id IS NOT NULL THEN
            SET error_message = CONCAT('The item "', item_name, '" has expired. Company: ', company_name, ', Email: ', company_email);
            SET error_id = fn_insert_into_error_message(error_message);
            INSERT INTO ItemLog (ItemID, CompanyID, LogTime, ErrorID)
            VALUES (item_id, company_id,NOW(), error_id);
        END IF;
    END LOOP;

    CLOSE expired_items;
END$$
DELIMITER ;


#--------------------------------#
-- To See and Test Event
#--------------------------------#


INSERT INTO Item (Name, Category, StorageType, ExpirationDate)
VALUE
    ('Salmon', 'Fresh Meat', 'Ambient', '2025-01-18 12:00:00');
SELECT
    il.*,
    em.ErrorMessage AS ErrorMessages
FROM
    ItemLog il
LEFT JOIN
    ErrorMessages em
ON
    il.ErrorID = em.ErrorID;