drop database Aisle_Management;

CREATE DATABASE Aisle_Management;
Use Aisle_Management;

#--------------------------------#
-- For Tables
#--------------------------------#

-- Create Aisle table
CREATE TABLE Aisle (
    AisleID INT PRIMARY KEY,
    Name VARCHAR(255) NOT NULL
);

-- Create Item table
CREATE TABLE Item (
    ItemID INT PRIMARY KEY auto_increment,
    Name VARCHAR(255) NOT NULL,
    Category VARCHAR(255),
    StorageType VARCHAR(255),
    ExpirationDate DATE
);

-- Create Company table
CREATE TABLE Company (
    CompanyID INT PRIMARY KEY,
    Name VARCHAR(255) NOT NULL,
    Email VARCHAR(255),
    Location VARCHAR(255)
);

-- Create the Contains Table
CREATE TABLE Contains (
    AisleID INT NOT NULL auto_increment,
    ItemID INT NOT NULL,
    PRIMARY KEY (AisleID, ItemID),
    FOREIGN KEY (AisleID) REFERENCES Aisle(AisleID),
    FOREIGN KEY (ItemID) REFERENCES Item(ItemID)
);

-- Create the Manufactured_By Table
CREATE TABLE Manufactured_By (
    ItemID INT NOT NULL auto_increment,
    CompanyID INT NOT NULL,
    PRIMARY KEY (ItemID, CompanyID),
    FOREIGN KEY (ItemID) REFERENCES Item(ItemID),
    FOREIGN KEY (CompanyID) REFERENCES Company(CompanyID)
);

-- Create ErrorMessages table
CREATE TABLE ErrorMessages (
    ErrorID INT PRIMARY KEY auto_increment,
    ErrorMessage VARCHAR(255) NOT NULL
);

-- Create ItemLog table
CREATE TABLE ItemLog (
    LogID INT PRIMARY KEY auto_increment,
    ItemID INT,
    CompanyID INT,
    LogTime DATETIME,
    ErrorID INT,
    FOREIGN KEY (ItemID) REFERENCES Item(ItemID),
    FOREIGN KEY (CompanyID) REFERENCES Company(CompanyID),
    FOREIGN KEY (ErrorID) REFERENCES ErrorMessages(ErrorID)
);

-- Create ItemLogErrors table
CREATE TABLE ItemLogErrors (
    LogID INT PRIMARY KEY auto_increment,
    ItemID INT,
    AisleID INT,
    LogTime DATETIME,
    ErrorID INT,
    FOREIGN KEY (ItemID) REFERENCES Item(ItemID),
    FOREIGN KEY (AisleID) REFERENCES Aisle(AisleID),
    FOREIGN KEY (ErrorID) REFERENCES ErrorMessages(ErrorID)
);


#--------------------------------#
-- For Functions and Triggers
#--------------------------------#


DELIMITER $$
CREATE TRIGGER trg_check_item_aisle_count
BEFORE INSERT ON Contains
FOR EACH ROW
BEGIN
    DECLARE aisle_count INT;

    -- Check if the ItemID already exists in another aisle (excluding the same aisle)
    SELECT COUNT(*) INTO aisle_count
    FROM Contains
    WHERE ItemID = NEW.ItemID AND AisleID != NEW.AisleID;

    -- If the item already exists in another aisle, raise an error
    IF aisle_count > 0 THEN
        SIGNAL SQLSTATE '45000'
        SET MESSAGE_TEXT = 'An item can only belong to one aisle.';
    END IF;
END$$
DELIMITER ;


DELIMITER $$
CREATE TRIGGER EnsureSingleManufacturer
BEFORE INSERT ON Manufactured_By
FOR EACH ROW
BEGIN
    -- Check if the ItemID already exists in the Manufactured_By table
    IF EXISTS (
        SELECT 1
        FROM Manufactured_By
        WHERE ItemID = NEW.ItemID
    ) THEN
        SIGNAL SQLSTATE '45000'
        SET MESSAGE_TEXT = 'Each Item can only be manufactured by one Company.';
    END IF;
END$$
DELIMITER ;


DELIMITER $$
CREATE FUNCTION fn_validate_aisle_compliance(item_id INT, aisle_id INT)
RETURNS VARCHAR(255)
DETERMINISTIC
BEGIN
    DECLARE item_storage_type VARCHAR(50);
    DECLARE item_category VARCHAR(50);
    DECLARE error_message VARCHAR(255);

    -- Fetch storage type and category
    SELECT StorageType, Category INTO item_storage_type, item_category
    FROM Item
    WHERE ItemID = item_id;

    -- Initialize error message
    SET error_message = NULL;

    -- Validate based on storage type and category
    CASE
        -- Frozen storage type validation
        WHEN item_storage_type = 'Frozen' AND
             (aisle_id <> (SELECT AisleID FROM Aisle WHERE Name = 'Frozen') OR
              item_category NOT IN ('Frozen', 'Fresh Meat')) THEN
            SET error_message = CONCAT(item_category, ' items with Frozen storage must be placed in the Frozen aisle.');

        -- Refrigerated storage type validation
        WHEN item_storage_type = 'Refrigerated' AND
             item_category IN ('Baked', 'Fresh Meat', 'Beverages') AND
             aisle_id <> (SELECT AisleID FROM Aisle WHERE Name = 'Refrigerated') THEN
            SET error_message = CONCAT(item_category, ' items with Refrigerated storage must be placed in the Refrigerated aisle.');

        -- Ambient storage type validation
        WHEN item_storage_type = 'Ambient' AND
             item_category IN ('Refrigerated', 'Frozen') THEN
            SET error_message = CONCAT(item_category, ' items cannot have Ambient storage type.');

        -- Vegetables validation
        WHEN item_category = 'Vegetables' AND
             aisle_id IN (SELECT AisleID FROM Aisle WHERE Name IN ('Frozen', 'Refrigerated')) THEN
            SET error_message = 'Vegetables cannot be placed in Frozen or Refrigerated aisles.';

        -- Kitchenware validation
        WHEN item_category = 'Kitchenware' AND
             aisle_id <> (SELECT AisleID FROM Aisle WHERE Name = 'Kitchenware') THEN
            SET error_message = 'Kitchenware items must be placed in the Kitchenware aisle.';

        -- Pet Food validation
        WHEN item_category = 'Pet Food' AND
             aisle_id NOT IN (SELECT AisleID FROM Aisle WHERE Name IN ('Pet Food', 'Pharmacy')) THEN
            SET error_message = 'Pet Food must be placed in the Pet Food or Pharmacy aisle.';

        -- Dry Food, Baked, Beverages, and Snacks validation
        WHEN item_category IN ('Baked', 'Beverages') AND
             item_storage_type = 'Frozen' THEN
            SET error_message = CONCAT(item_category, ' items cannot be stored in Frozen aisles.');

        WHEN item_category in ('Dry Food', 'Snacks') AND item_storage_type
                in ('Frozen', 'Refrigerated') THEN
            SET error_message = CONCAT(item_category, ' items cannot be stored in Frozen/Refrigerated aisles.');

        -- Catch-all for unexpected cases
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
    SELECT StorageType, Category INTO item_storage_type, item_category
    FROM Item
    WHERE ItemID = item_id;

    -- Determine the correct aisle
    CASE
        WHEN item_storage_type = 'Frozen' THEN
            SELECT AisleID INTO suggested_aisle FROM Aisle WHERE Name = 'Frozen';
        WHEN item_storage_type = 'Refrigerated' THEN
            SELECT AisleID INTO suggested_aisle FROM Aisle WHERE Name = 'Refrigerated';
        WHEN item_category = 'Vegetables' THEN
            SELECT AisleID INTO suggested_aisle FROM Aisle WHERE Name = 'Vegetables';
        WHEN item_category = 'Kitchenware' THEN
            SELECT AisleID INTO suggested_aisle FROM Aisle WHERE Name = 'Kitchenware';
        WHEN item_category = 'Pet Food' THEN
            SELECT AisleID INTO suggested_aisle FROM Aisle WHERE Name IN ('Pet Food', 'Pharmacy') LIMIT 1;
        ELSE
            SELECT AisleID INTO suggested_aisle FROM Aisle WHERE Name = 'Ambient';
    END CASE;

    RETURN suggested_aisle;
END$$
DELIMITER ;


DELIMITER $$
CREATE PROCEDURE pr_insert_item_log(item_id INT, aisle_id INT, error_message VARCHAR(255))
BEGIN
    -- Insert the log entry into the ItemLog table
    INSERT INTO ItemLogErrors (ItemID, AisleID, LogTime, ErrorID)
    VALUES (item_id, aisle_id, NOW(),error_message);
END$$
DELIMITER ;


DELIMITER $$
CREATE FUNCTION fn_insert_into_error_message(error_message VARCHAR(255))
RETURNS INT
DETERMINISTIC
BEGIN
    DECLARE error_id INT;
    -- Insert the log entry into the ItemLog table
   IF NOT EXISTS (
            SELECT 1
            FROM ErrorMessages
            WHERE ErrorMessage = error_message
        ) THEN
            -- If not, insert the error message into the ErrorMessages table
            INSERT INTO ErrorMessages (ErrorMessage)
            VALUES (error_message);
        END IF;
    select ErrorID into error_id
        from ErrorMessages
        where ErrorMessage = error_message;
    RETURN error_id;
END$$
DELIMITER ;


DELIMITER $$
CREATE TRIGGER trg_log_item_wrong_aisle
AFTER INSERT ON Contains
FOR EACH ROW
BEGIN
    DECLARE error_message VARCHAR(255);
    DECLARE error_id INT;

    -- Call the validation function to check compliance
    set error_message = fn_validate_aisle_compliance(NEW.ItemID, NEW.AisleID);

    -- If non-compliant, suggest correct aisle, log the error, and reject the insertion
    IF error_message IS NOT NULL THEN
        -- Call the suggestion function to get the correct aisle

        SET error_id = fn_insert_into_error_message(error_message);
        -- Call the insertion procedure to log the incorrect insertion
        CALL pr_insert_item_log(NEW.ItemID,NEW.AisleID, error_id);

    END IF;
END$$
DELIMITER ;


#--------------------------------#
-- For Insertions
#--------------------------------#

INSERT INTO Aisle (AisleID, Name)
VALUES
    (1, 'Vegetables'),
    (2, 'Dry Food'),
    (3, 'Fresh Meat'),
    (4, 'Baked'),
    (5, 'Kitchenware'),
    (6, 'Pharmacy'),
    (7, 'Pet Food'),
    (8, 'Frozen'),
    (9, 'Refrigerated'),
    (10, 'Beverages'),
    (11, 'Snacks');

INSERT INTO Company (CompanyID, Name,Email, Location)
VALUES
    (1, 'Fresh Farms', 'company1@blabla.bla', 'France'),
    (2, 'Dry Goods Co.', 'company2@blabla.bla', 'Germany'),
    (3, 'Meat Masters', 'company3@blabla.bla', 'Italy'),
    (4, 'Bakers Delight', 'company4@blabla.bla', 'Spain'),
    (5, 'KitchenPro', 'company5@blabla.bla', 'Netherlands');

INSERT INTO Item (Name, Category, StorageType, ExpirationDate)
VALUES
    -- Aisle 1: Vegetables
    ('Lettuce', 'Vegetables', 'Ambient', '2025-01-20 12:00:00'),
    ('Tomatoes', 'Vegetables', 'Ambient', '2025-01-22 12:00:00'),
    ('Potatoes', 'Vegetables', 'Ambient', '2025-02-01 12:00:00'),

    -- Aisle 2: Dry Food
    ('Sugar', 'Dry Food', 'Ambient', '2025-12-31 12:00:00'),
    ('Coffee', 'Dry Food', 'Ambient', '2026-01-15 12:00:00'),
    ('Flour', 'Dry Food', 'Ambient', '2025-11-30 12:00:00'),

    -- Aisle 3: Fresh Meat
    ('Beef', 'Fresh Meat', 'Ambient', '2025-01-19 12:00:00'),
    ('Lamb', 'Fresh Meat', 'Ambient', '2025-01-20 12:00:00'),
    ('Salmon', 'Fresh Meat', 'Ambient', '2025-01-18 12:00:00'),

    -- Aisle 4: Baked
    ('Bread', 'Baked', 'Ambient', '2025-01-25 12:00:00'),
    ('Bagels', 'Baked', 'Ambient', '2025-01-24 12:00:00'),
    ('Cakes', 'Baked', 'Refrigerated', '2025-01-23 12:00:00'),

    -- Aisle 5: Kitchenware
    ('Cooking Pot', 'Kitchenware', 'Ambient', '2100-01-01 12:00:00'),
    ('Knife Set', 'Kitchenware', 'Ambient', '2100-01-01 12:00:00'),
    ('Blender', 'Kitchenware', 'Ambient', '2100-01-01 12:00:00'),

    -- Aisle 6: Pharmacy
    ('Aspirin', 'Pharmacy', 'Ambient', '2025-06-30 12:00:00'),
    ('Bandages', 'Pharmacy', 'Ambient', '2025-07-01 12:00:00'),
    ('Diet Pet Food', 'Pet Food', 'Ambient', '2025-05-30 12:00:00'),

    -- Aisle 7: Pet Food
    ('Dog Food', 'Pet Food', 'Ambient', '2025-04-30 12:00:00'),
    ('Cat Food', 'Pet Food', 'Ambient', '2025-04-30 12:00:00'),
    ('Fish Food', 'Pet Food', 'Ambient', '2025-04-30 12:00:00'),

    -- Aisle 8: Frozen
    ('Frozen Pizza', 'Frozen', 'Frozen', '2025-03-31 12:00:00'),
    ('Frozen Berries', 'Frozen', 'Frozen', '2025-03-31 12:00:00'),
    ('Frozen Beef', 'Fresh Meat', 'Frozen', '2025-03-31 12:00:00'),

    -- Aisle 9: Refrigerated
    ('Milk', 'Dairy', 'Refrigerated', '2025-01-21 12:00:00'),
    ('Yogurt', 'Dairy', 'Refrigerated', '2025-01-22 12:00:00'),
    ('Cheese', 'Dairy', 'Refrigerated', '2025-01-23 12:00:00'),

    -- Aisle 10: Beverages
    ('Cola', 'Beverages', 'Ambient', '2025-08-30 12:00:00'),
    ('Cold Cola', 'Beverages', 'Ambient', '2025-08-30 12:00:00'),
    ('Orange Juice', 'Beverages', 'Ambient', '2025-08-30 12:00:00'),

    -- Aisle 11: Snacks
    ('Chips', 'Snacks', 'Ambient', '2025-09-15 12:00:00'),
    ('Nuts', 'Snacks', 'Ambient', '2025-10-01 12:00:00'),
    ('Chocolate', 'Snacks', 'Ambient', '2025-09-20 12:00:00');

INSERT INTO Contains (AisleID, ItemID)
VALUES
    -- Aisle 1: Vegetables
    (1, 1), (1, 2), (1, 3),
    -- Aisle 2: Dry Food
    (2, 4), (2, 5), (2, 6),
    -- Aisle 3: Fresh Meat
    (3, 7), (3, 8), (3, 9),
    -- Aisle 4: Baked
    (4, 10), (4, 11), (4, 12),
    -- Aisle 5: Kitchenware
    (5, 13), (5, 14), (5, 15),
    -- Aisle 6: Pharmacy
    (6, 16), (6, 17), (6, 18),
    -- Aisle 7: Pet Food
    (7, 19), (7, 20), (7, 21),
    -- Aisle 8: Frozen (only frozen items should be here)
    (8, 22), (8, 23), (8, 24),
    -- Aisle 9: Refrigerated (only refrigerated items should be here)
    (9, 25), (9, 26), (9, 27),
    -- Aisle 10: Beverages
    (10, 28), (10, 29), (10, 30),
    -- Aisle 11: Snacks
    (11, 31), (11, 32), (11, 33);

INSERT INTO Manufactured_By (ItemID, CompanyID)
VALUES
    (1, 1), (2, 1), (3, 1),
    (4, 2), (5, 2), (6, 2),
    (7, 3), (8, 3), (9, 3),
    (10, 4), (11, 4), (12, 4),
    (13, 5), (14, 5), (15, 5),
    (16, 1), (17, 1), (18, 1),
    (19, 2), (20, 2), (21, 2),
    (22, 3), (23, 3), (24, 3),
    (25, 4), (26, 4), (27, 4),
    (28, 5), (29, 5), (30, 5),
    (31, 1), (32, 1), (33, 1);

#--------------------------------#
-- For Testing Triggers
#--------------------------------#

INSERT INTO Item
value (34,'Canned beef','Dry food','Refrigerated','2025-02-01');

insert into contains
    value (2,34);

select * from ItemLogErrors;
show triggers;

#--------------------------------#
-- For Joins
#--------------------------------#

-- Inner Join
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
-- For Stored Procedures
#--------------------------------#

DELIMITER $$
CREATE PROCEDURE CheckItemCompliance(IN item_id INT, IN aisle_id INT)
BEGIN
    DECLARE item_storage_type VARCHAR(50);
    DECLARE item_category VARCHAR(50);
    DECLARE error_message VARCHAR(255);
    DECLARE suggested_aisle INT;

    -- Fetch storage type and category for the item
    SELECT StorageType, Category INTO item_storage_type, item_category
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