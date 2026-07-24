drop database Aisle_Management;

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
    FOREIGN KEY (SuperMarketID) REFERENCES SuperMarket(SuperMarketID)
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

CREATE TRIGGER trg_check_Item_Aisle_count
BEFORE INSERT ON Contain
FOR EACH ROW
BEGIN
    DECLARE Aisle_count INT;

    SELECT COUNT(*) INTO Aisle_count
    FROM Contain
    WHERE ItemID = NEW.ItemID 
      AND SuperMarketID = NEW.SuperMarketID 
      AND AisleID != NEW.AisleID;

    IF Aisle_count > 0 THEN
        SIGNAL SQLSTATE '45000'
        SET MESSAGE_TEXT = 'An Item can only belong to one Aisle per Supermarket.';
    END IF;
END;

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

CREATE FUNCTION fn_suggest_correct_aisle(item_id INT, supermarket_id INT)
RETURNS INT
DETERMINISTIC
BEGIN
    DECLARE item_storage_type VARCHAR(50);
    DECLARE item_category VARCHAR(50);
    DECLARE suggested_aisle INT;

    -- Fetch item details
    SELECT ItemStorageType, ItemCategory
    INTO item_storage_type, item_category
    FROM Item
    WHERE ItemID = item_id;

    -- Match aisle by name within the correct supermarket
    IF item_storage_type = 'Frozen' THEN
        SELECT AisleID INTO suggested_aisle
        FROM Aisle
        WHERE AisleName = 'Frozen' AND SuperMarketID = supermarket_id
        LIMIT 1;

    ELSEIF item_storage_type = 'Refrigerated' THEN
        SELECT AisleID INTO suggested_aisle
        FROM Aisle
        WHERE AisleName = 'Refrigerated' AND SuperMarketID = supermarket_id
        LIMIT 1;

    ELSEIF item_category = 'Vegetables' THEN
        SELECT AisleID INTO suggested_aisle
        FROM Aisle
        WHERE AisleName = 'Vegetables' AND SuperMarketID = supermarket_id
        LIMIT 1;

    ELSEIF item_category = 'Kitchenware' THEN
        SELECT AisleID INTO suggested_aisle
        FROM Aisle
        WHERE AisleName = 'Kitchenware' AND SuperMarketID = supermarket_id
        LIMIT 1;

    ELSEIF item_category = 'Pet Food' THEN
        SELECT AisleID INTO suggested_aisle
        FROM Aisle
        WHERE AisleName IN ('Pet Food', 'Pharmacy') AND SuperMarketID = supermarket_id
        LIMIT 1;

    ELSE
        SELECT AisleID INTO suggested_aisle
        FROM Aisle
        WHERE AisleName = 'Ambient' AND SuperMarketID = supermarket_id
        LIMIT 1;
    END IF;

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
    DECLARE maxdistance FLOAT DEFAULT 7;
    DECLARE checknexpiry DATE;
    DECLARE nexpiry BOOLEAN;
    DECLARE nproducerid INT;
    DECLARE nerrorid INT;
    DECLARE ndistance FLOAT;
    DECLARE nperishable BOOLEAN;
    DECLARE error_message VARCHAR(255);

    -- Use SuperMarketID directly from NEW.SuperMarketID
    SELECT ProducerID INTO nproducerid
    FROM Manufactured_By
    WHERE ItemID = NEW.ItemID
    LIMIT 1;

    SELECT Distance INTO ndistance
    FROM Distance
    WHERE SuperMarketID = NEW.SuperMarketID AND ProducerID = nproducerid
    LIMIT 1;

    SELECT ItemPerishable, ItemExpirationDate INTO nperishable, checknexpiry
    FROM Item
    WHERE ItemID = NEW.ItemID;

    SET nexpiry = (checknexpiry < NOW());

    IF nexpiry THEN
        IF ndistance <= maxdistance OR nperishable = 0 THEN
            SET error_message = 'Expired but returnable due to being non perishable or short distance to producer';
            SET nerrorid = fn_insert_into_error_message(error_message);

            INSERT INTO ItemLogErrors(ItemID, AisleID, ErrorID, LogTime, ToBeThrown)
            VALUES (NEW.ItemID, NEW.AisleID, nerrorid, NOW(), FALSE);
        ELSE
            SET error_message = 'Expired and to be thrown away';
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

#--------------------------------#
-- Example of wrong aisle placement
#--------------------------------#

#INSERT INTO Contain (AisleID, ItemID, SuperMarketID) VALUES (6, 4, 1);


#--------------------------------#
-- Joins
#--------------------------------#

#-----------------------------------------------#
-- Show only The items and on which aisle they are found on
#-----------------------------------------------#

SELECT i.ItemID, i.ItemName AS ItemName, a.AisleName AS AisleName
FROM Item i
INNER JOIN Contain c ON i.ItemID = c.ItemID
INNER JOIN Aisle a ON c.AisleID = a.AisleID;

#-----------------------------------------------#
-- View of all items on all aisle for every Supermarket on the Database
#-----------------------------------------------#

#-----------------------------------------------#
-- View of the errors produced
#-----------------------------------------------#

CREATE VIEW FullItemDetails AS
SELECT 
    i.ItemID, i.ItemName, i.ItemCategory, i.ItemPerishable,
    p.ProducerID, p.ProducerName,
    s.SuperMarketID, s.SuperMarketName,
    a.AisleID, a.AisleName
FROM Item i
JOIN Manufactured_By mb ON i.ItemID = mb.ItemID
JOIN Producer p ON mb.ProducerID = p.ProducerID
JOIN Contain c ON i.ItemID = c.ItemID
JOIN Aisle a ON c.AisleID = a.AisleID
JOIN SuperMarket s ON a.SuperMarketID = s.SuperMarketID;

SELECT * FROM FullItemDetails;

#-----------------------------------------------#
-- View of only where the items come from
#-----------------------------------------------#

CREATE VIEW ItemWithProducers AS
SELECT 
    i.ItemID, i.ItemName, i.ItemCategory, i.ItemPerishable,
    p.ProducerID, p.ProducerName, p.ProducerLocation
FROM Item i
JOIN Manufactured_By mb ON i.ItemID = mb.ItemID
JOIN Producer p ON mb.ProducerID = p.ProducerID
ORDER BY p.ProducerID;

SELECT * FROM ItemWithProducers;

#-----------------------------------------------#
-- View of the distance of supermarket to producer (or viceversa)
#-----------------------------------------------#

CREATE VIEW ProducerSuperMarketDistance AS
SELECT 
    p.ProducerID, p.ProducerName,
    s.SuperMarketID, s.SuperMarketName,
    d.Distance
FROM Distance d
JOIN Producer p ON d.ProducerID = p.ProducerID
JOIN SuperMarket s ON d.SuperMarketID = s.SuperMarketID;

SELECT * FROM ProducerSuperMarketDistance;

#-----------------------------------------------#
-- View of Items by Storage Type and Aisle
#-----------------------------------------------#

CREATE VIEW WhereToStore AS
SELECT i.ItemStorageType, a.AisleName AS AisleName, i.ItemName AS ItemName
FROM Item i
JOIN Contain c ON i.ItemID = c.ItemID
JOIN Aisle a ON c.AisleID = a.AisleID
ORDER BY i.ItemStorageType, a.AisleName;

SELECT * FROM WhereToStore;

#-----------------------------------------------#
-- View of the errors produced
#-----------------------------------------------#

CREATE VIEW ItemErrorDetails AS
SELECT 
    le.LogID, le.LogTime, le.ToBeThrown,
    i.ItemID, i.ItemName,
    a.AisleID, a.AisleName,
    em.ErrorMessage
FROM ItemLogErrors le
JOIN Item i ON le.ItemID = i.ItemID
JOIN Aisle a ON le.AisleID = a.AisleID
LEFT JOIN ErrorMessages em ON le.ErrorID = em.ErrorID;

SELECT * FROM ItemErrorDetails;


#--------------------------------#
-- Stored Procedures
#--------------------------------#

#--------------------------------#
-- Add Item To Aisle
#--------------------------------#

DELIMITER $$

CREATE PROCEDURE AddItemToAisle (
    IN inItemID INT,
    IN inAisleID INT,
    IN inSuperMarketID INT
)
BEGIN
    -- Check that the aisle belongs to the supermarket
    IF EXISTS (
        SELECT 1 FROM Aisle
        WHERE AisleID = inAisleID AND SuperMarketID = inSuperMarketID
    ) THEN
        -- Insert into Contain table
        INSERT INTO Contain (AisleID, ItemID, SuperMarketID)
        VALUES (inAisleID, inItemID, inSuperMarketID);
    ELSE
        SIGNAL SQLSTATE '45000'
        SET MESSAGE_TEXT = 'Aisle does not belong to the specified supermarket.';
    END IF;
END $$

DELIMITER ;

#--------------------------------#
-- Remove Item From SuperMarket
#--------------------------------#

DELIMITER $$

CREATE PROCEDURE RemoveItemFromSuperMarket (
    IN inItemID INT,
    IN inSuperMarketID INT
)
BEGIN
    DELETE FROM Contain
    WHERE ItemID = inItemID AND SuperMarketID = inSuperMarketID;
END $$

DELIMITER ;

#--------------------------------#
-- Manually create/add errors
#--------------------------------#

DELIMITER $$

CREATE PROCEDURE LogItemError (
    IN inItemID INT,
    IN inAisleID INT,
    IN inErrorMessage VARCHAR(255)
)
BEGIN
    DECLARE errID INT;

    -- Insert error message if not already present
    INSERT INTO ErrorMessages (ErrorMessage)
    VALUES (inErrorMessage);

    -- Retrieve the ErrorID
    SELECT ErrorID INTO errID
    FROM ErrorMessages
    WHERE ErrorMessage = inErrorMessage;

    -- Insert log entry
    INSERT INTO ItemLogErrors (ItemID, AisleID, ErrorID, LogTime, ToBeThrown)
    VALUES (inItemID, inAisleID, errID, NOW(), TRUE);
END $$

DELIMITER ;

#--------------------------------#
-- Remove Item after expiration
#--------------------------------#

DELIMITER $$

CREATE PROCEDURE CleanExpiredItems ()
BEGIN
    DELETE FROM Contain
    WHERE ItemID IN (
        SELECT ItemID FROM Item
        WHERE ItemPerishable = TRUE AND ItemExpirationDate < CURDATE()
    );
END $$

DELIMITER ;

#--------------------------------#
-- Check if the Item complies with Database structure
#--------------------------------#

DELIMITER $$

CREATE PROCEDURE CheckItemCompliance(
    IN item_id INT,
    IN aisle_id INT,
    IN supermarket_id INT
)
BEGIN
    DECLARE compliance_message VARCHAR(255) DEFAULT NULL;
    DECLARE suggested_aisle INT;


    -- Call your validation function (assumes it returns BOOLEAN or 0/1)
    SET compliance_message = fn_validate_aisle_compliance(item_id, aisle_id, supermarket_id);

    IF compliance_message IS NULL THEN
        SELECT 'Aisle is compliant.' AS Message;
    ELSE
        -- Suggest correct aisle (assume function returns INT AisleID)
        SET suggested_aisle = fn_suggest_correct_aisle(item_id, supermarket_id);

        SELECT compliance_message AS ErrorMessage, suggested_aisle AS SuggestedAisleID;
    END IF;
END $$

DELIMITER ;

#--------------------------------#
-- Check if the Item complies with Database structure if not, output an error and aisle reccomendation, then log it into the ItemlogError table
#--------------------------------#

DELIMITER $$

CREATE PROCEDURE sp_check_item_placement()
BEGIN
    DECLARE item_id INT;
    DECLARE aisle_id INT;
    DECLARE item_storage_type VARCHAR(50);
    DECLARE item_category VARCHAR(50);
    DECLARE aisle_name VARCHAR(50);
    DECLARE suggested_aisle INT;
    DECLARE error_message VARCHAR(255);
    DECLARE error_id INT;
    DECLARE supermarket_id INT;

    DECLARE done INT DEFAULT 0;

    DECLARE cur CURSOR FOR
        SELECT I.ItemID, I.ItemStorageType, I.ItemCategory, A.AisleName, C.AisleID, C.SuperMarketID
        FROM Item I
        JOIN Contain C ON I.ItemID = C.ItemID
        JOIN Aisle A ON C.AisleID = A.AisleID;

    DECLARE CONTINUE HANDLER FOR NOT FOUND SET done = 1;

    OPEN cur;

    read_loop: LOOP
        FETCH cur INTO item_id, item_storage_type, item_category, aisle_name, aisle_id, supermarket_id;

        IF done THEN
            LEAVE read_loop;
        END IF;

        SET suggested_aisle = fn_suggest_correct_aisle(item_id, supermarket_id);

        IF aisle_id <> suggested_aisle THEN
            SET error_message = CONCAT(item_category, ' item is not in the suggested aisle: ', suggested_aisle);
        END IF;

        IF error_message IS NOT NULL THEN
            SET error_id = fn_insert_into_error_message(error_message);
            INSERT INTO ItemLogErrors (ItemID, AisleID, LogTime, ErrorID)
            VALUES (item_id, aisle_id, NOW(), error_id);
        END IF;

        -- Reset for next loop
        SET error_message = NULL;
    END LOOP;

    CLOSE cur;
END $$

DELIMITER ;

#--------------------------------#
-- To Use Stored Procedures
#--------------------------------#


#--------------------------------#
-- Checks is the item is in the correct aisle if not suggest where to place it
#--------------------------------#
call CheckItemCompliance(2, 8, 1);

call sp_check_item_placement();

SELECT
    em.ErrorMessage,
    i.ItemName, i.ItemStorageType,
    a.AisleID, a.AisleName AS IncorrectAisle,
    le.LogTime,
    le.ToBeThrown
FROM ItemLogErrors le
JOIN Item i ON le.ItemID = i.ItemID
JOIN Aisle a ON le.AisleID = a.AisleID
LEFT JOIN ErrorMessages em ON le.ErrorID = em.ErrorID
ORDER BY le.LogTime DESC;

# ----------------------------------
-- Sanity check: run daily to validate item placement and log errors
# ----------------------------------

DELIMITER$$

CREATE EVENT ev_daily_item_placement_check
ON SCHEDULE EVERY 1 DAY
STARTS CURRENT_TIMESTAMP
DO
BEGIN
    CALL sp_check_item_placement();
END $$

DELIMITER ;

# --------------------------------------------
-- Logs expired perishable items from the Contain table daily
# --------------------------------------------

DELIMITER $$

CREATE PROCEDURE sp_expiration_check()
BEGIN
    DECLARE done INT DEFAULT 0;
    DECLARE item_id INT;
    DECLARE item_name VARCHAR(100);
    DECLARE item_expiration DATE;
    DECLARE producer_id INT;
    DECLARE producer_name VARCHAR(100);
    DECLARE producer_email VARCHAR(255);
    DECLARE error_message VARCHAR(255);
    DECLARE error_id INT;

    DECLARE expired_items CURSOR FOR
        SELECT ItemID, ItemName, ItemExpirationDate
        FROM Item
        WHERE ItemPerishable = TRUE AND ItemExpirationDate <= CURDATE();

    DECLARE CONTINUE HANDLER FOR NOT FOUND SET done = 1;

    OPEN expired_items;

    read_loop: LOOP
        FETCH expired_items INTO item_id, item_name, item_expiration;
        IF done THEN
            LEAVE read_loop;
        END IF;

        SELECT mb.ProducerID, p.ProducerName, p.ProducerLocation
        INTO producer_id, producer_name, producer_email
        FROM Manufactured_By mb
        JOIN Producer p ON mb.ProducerID = p.ProducerID
        WHERE mb.ItemID = item_id
        LIMIT 1;

        IF producer_id IS NOT NULL THEN
            SET error_message = CONCAT('The item "', item_name, '" has expired. Producer: ', producer_name, ', Location: ', producer_email);
            SET error_id = fn_insert_into_error_message(error_message);

            INSERT INTO ItemLogErrors (ItemID, AisleID, LogTime, ErrorID)
            VALUES (item_id, NULL, NOW(), error_id);
        END IF;
    END LOOP;

    CLOSE expired_items;
END $$

DELIMITER ;

# --------------------------------------------
-- Removes expired perishable items from the Contain table daily
# --------------------------------------------

DELIMITER $$

CREATE EVENT ev_daily_expiration_and_cleanup
ON SCHEDULE EVERY 1 DAY
STARTS CURRENT_TIMESTAMP
DO
BEGIN
    CALL sp_expiration_check();
    CALL CleanExpiredItems();
END $$

DELIMITER ;


#--------------------------------#
-- For Event Handling
#--------------------------------#

DELIMITER $$

CREATE EVENT ev_daily_expiration_process
ON SCHEDULE EVERY 1 DAY
STARTS CURRENT_TIMESTAMP
DO
BEGIN
    CALL sp_expiration_check();
    CALL ev_daily_expiration_and_cleanup();
END $$

DELIMITER ;

#--------------------------------#
-- To See and Test Event
#--------------------------------#


INSERT INTO Item (ItemID, ItemName, ItemCategory, ItemStorageType, ItemPerishable, ItemExpirationDate)
VALUE
    (11, 'Salmon', 'Fresh Meat', 'Ambient', TRUE, '2025-01-18');

SELECT
    il.*,
    em.ErrorMessage AS ErrorMessages
FROM
    ItemLogErrors il
LEFT JOIN
    ErrorMessages em
ON
    il.ErrorID = em.ErrorID;