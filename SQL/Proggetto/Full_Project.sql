DROP DATABASE IF EXISTS dimensional_transfer;
CREATE DATABASE dimensional_transfer;
USE dimensional_transfer;

-- Create tables
DROP TABLE IF EXISTS item;
CREATE TABLE Item (
    item_id INT AUTO_INCREMENT PRIMARY KEY,
    item_name VARCHAR(255) NOT NULL
);

DROP TABLE IF EXISTS Guild;
CREATE TABLE Guild (
    guild_id INT AUTO_INCREMENT PRIMARY KEY,
    guild_name VARCHAR(255) NOT NULL DEFAULT 'none'
);

DROP TABLE IF EXISTS Player;
CREATE TABLE Player (
    player_id INT AUTO_INCREMENT PRIMARY KEY,
    player_name VARCHAR(255) NOT NULL,
    guild_id INT,
    FOREIGN KEY (guild_id) REFERENCES Guild(guild_id)
);

DROP TABLE IF EXISTS NPC;
CREATE TABLE NPC (
    npc_id INT AUTO_INCREMENT PRIMARY KEY,
    npc_name VARCHAR(255) NOT NULL,
    guild_id INT,
    FOREIGN KEY (guild_id) REFERENCES Guild(guild_id)
);

DROP TABLE IF EXISTS Player_Item;
CREATE TABLE Player_Item (
    player_item_id INT AUTO_INCREMENT PRIMARY KEY,
    player_id INT,
    item_id INT,
    state BOOLEAN DEFAULT TRUE,
    FOREIGN KEY (player_id) REFERENCES Player(player_id),
    FOREIGN KEY (item_id) REFERENCES Item(item_id)
);

DROP TABLE IF EXISTS Quest;
CREATE TABLE Quest (
    quest_id INT AUTO_INCREMENT PRIMARY KEY,
    quest_name VARCHAR(255) NOT NULL,
    state BOOLEAN DEFAULT TRUE
);

DROP TABLE IF EXISTS Dimension;
CREATE TABLE Dimension (
    dimension_id INT AUTO_INCREMENT PRIMARY KEY,
    dimension_name VARCHAR(255) NOT NULL
);

DROP TABLE IF EXISTS Achievement;
CREATE TABLE Achievement (
    achievement_id INT AUTO_INCREMENT PRIMARY KEY,
    achievement_name VARCHAR(255) NOT NULL
);

DROP TABLE IF EXISTS Complete;
CREATE TABLE Complete (
    player_id INT,
    quest_id INT,
    PRIMARY KEY (player_id, quest_id),
    FOREIGN KEY (player_id) REFERENCES Player(player_id),
    FOREIGN KEY (quest_id) REFERENCES Quest(quest_id)
);

DROP TABLE IF EXISTS Talks;
CREATE TABLE Talks (
    player_id INT,
    npc_id INT,
    PRIMARY KEY (player_id, npc_id),
    FOREIGN KEY (player_id) REFERENCES Player(player_id),
    FOREIGN KEY (npc_id) REFERENCES NPC(npc_id)
);

DROP TABLE IF EXISTS Initiate;
CREATE TABLE Initiate (
    npc_id INT,
    quest_id INT,
    PRIMARY KEY (npc_id, quest_id),
    FOREIGN KEY (npc_id) REFERENCES NPC(npc_id),
    FOREIGN KEY (quest_id) REFERENCES Quest(quest_id)
);

DROP TABLE IF EXISTS Check_Achievement;
CREATE TABLE Check_Achievement (
    quest_id INT,
    player_id INT,
    requires_all_player_items BOOLEAN NOT NULL,
    PRIMARY KEY (quest_id, player_id),
    FOREIGN KEY (quest_id) REFERENCES Quest(quest_id),
    FOREIGN KEY (player_id) REFERENCES Player(player_id)
);

DROP TABLE IF EXISTS Affiliated;
CREATE TABLE Affiliated (
    guild_id INT,
    npc_id INT,
    affiliation VARCHAR(10),
    PRIMARY KEY (guild_id, npc_id),
    FOREIGN KEY (guild_id) REFERENCES Guild(guild_id),
    FOREIGN KEY (npc_id) REFERENCES NPC(npc_id)
);

DROP TABLE IF EXISTS Belongs;
CREATE TABLE Belongs (
    player_id INT,
    guild_id INT,
    PRIMARY KEY (player_id, guild_id),
    FOREIGN KEY (player_id) REFERENCES Player(player_id),
    FOREIGN KEY (guild_id) REFERENCES Guild(guild_id)
);

DROP TABLE IF EXISTS Owns;
CREATE TABLE Owns (
    player_id INT,
    player_item_id INT,
    PRIMARY KEY (player_id, player_item_id),
    FOREIGN KEY (player_id) REFERENCES Player(player_id),
    FOREIGN KEY (player_item_id) REFERENCES Player_Item(player_item_id)
);

DROP TABLE IF EXISTS Unlocks;
CREATE TABLE Unlocks (
    achievement_id INT,
    dimension_id INT,
    PRIMARY KEY (achievement_id, dimension_id),
    FOREIGN KEY (achievement_id) REFERENCES Achievement(achievement_id),
    FOREIGN KEY (dimension_id) REFERENCES Dimension(dimension_id)
);

DROP TABLE IF EXISTS Travel;
CREATE TABLE Travel (
    player_id INT,
    dimension_id INT,
    PRIMARY KEY (player_id, dimension_id),
    FOREIGN KEY (player_id) REFERENCES Player(player_id),
    FOREIGN KEY (dimension_id) REFERENCES Dimension(dimension_id)
);

-- Procedures

DROP PROCEDURE IF EXISTS InsertPlayerItem;
DELIMITER //
CREATE PROCEDURE InsertPlayerItem(p_player_id INT, p_item_id INT)
BEGIN
    IF NOT EXISTS (SELECT 1 FROM Item WHERE item_id = p_item_id) THEN
        UPDATE Player_Item SET state = FALSE WHERE player_id = p_player_id;
    ELSE
        INSERT INTO Player_Item (player_id, item_id, state) VALUES (p_player_id, p_item_id, TRUE);
    END IF;
END;
//
DELIMITER //

DROP PROCEDURE IF EXISTS CompleteQuests;
DELIMITER //
CREATE PROCEDURE CompleteQuests(IN p_player_id INT, IN p_quest_ids TEXT)
BEGIN
    DECLARE quest_id INT;
    DECLARE done BOOLEAN DEFAULT FALSE;
    DECLARE cur CURSOR FOR
        SELECT CAST(SUBSTRING_INDEX(SUBSTRING_INDEX(p_quest_ids, ',', numbers.n), ',', -1) AS UNSIGNED) AS quest_id
        FROM (SELECT @rownum := @rownum + 1 AS n FROM
            (SELECT 1 UNION SELECT 2 UNION SELECT 3 UNION SELECT 4 UNION SELECT 5 UNION SELECT 6 UNION SELECT 7 UNION SELECT 8 UNION SELECT 9 UNION SELECT 10) t
            CROSS JOIN (SELECT @rownum := 0) r
        ) numbers
        WHERE n <= CHAR_LENGTH(p_quest_ids) - CHAR_LENGTH(REPLACE(p_quest_ids, ',', '')) + 1;
    DECLARE CONTINUE HANDLER FOR NOT FOUND SET done = TRUE;

    OPEN cur;
    read_loop: LOOP
        FETCH cur INTO quest_id;
        IF done THEN
            LEAVE read_loop;
        END IF;
        INSERT IGNORE INTO Complete (player_id, quest_id) VALUES (p_player_id, quest_id);
    END LOOP;
    CLOSE cur;

    -- Update Check_Achievement status
    UPDATE Check_Achievement ca
    SET requires_all_player_items = (
        NOT EXISTS (SELECT 1 FROM Player_Item WHERE player_id = p_player_id AND state = FALSE)
        AND NOT EXISTS (SELECT 1 FROM Quest q LEFT JOIN Complete c ON q.quest_id = c.quest_id WHERE c.player_id = p_player_id AND q.state = TRUE AND c.quest_id IS NULL)
    )
    WHERE player_id = p_player_id;

    -- Output the completed quests
    SELECT * FROM Complete WHERE player_id = p_player_id;
END;
//
DELIMITER //

DROP PROCEDURE IF EXISTS CheckDimensionUnlock;
DELIMITER //
CREATE PROCEDURE CheckDimensionUnlock(IN p_player_id INT)
BEGIN
    DECLARE current_dimension_id INT;
    DECLARE next_dimension_id INT;
    DECLARE relevant_quests_completed BOOLEAN;

    SELECT dimension_id INTO current_dimension_id FROM Travel WHERE player_id = p_player_id;
    SET next_dimension_id = current_dimension_id + 1;

    SELECT COUNT(*) = 0 INTO relevant_quests_completed
    FROM Quest q
    JOIN Check_Achievement ca ON q.quest_id = ca.quest_id
    WHERE ca.requires_all_player_items = TRUE
    AND q.quest_id NOT IN (SELECT quest_id FROM Complete WHERE player_id = p_player_id);

    IF relevant_quests_completed THEN
        SET @unlock_allowed = TRUE;
    ELSE
        SET @unlock_allowed = FALSE;
    END IF;
END;
//
DELIMITER //

DROP PROCEDURE IF EXISTS ChangeDimension;
DELIMITER //
CREATE PROCEDURE ChangeDimension(IN p_player_id INT, IN new_dimension_id INT)
BEGIN
    DECLARE current_dimension_id INT;

    SELECT dimension_id INTO current_dimension_id FROM Travel WHERE player_id = p_player_id;
    IF new_dimension_id = current_dimension_id + 1 THEN
        UPDATE Travel SET dimension_id = new_dimension_id WHERE player_id = p_player_id;
        SELECT 'Move to successive dimension OK' AS message;
    ELSE
        SELECT 'Skipping dimension not allowed' AS message;
    END IF;
END;
//
DELIMITER //

DROP PROCEDURE IF EXISTS Test_InsertPlayerItem;
DELIMITER //
CREATE PROCEDURE Test_InsertPlayerItem(in who_to_check int)
BEGIN
    declare temp_value int;
    -- Show the state of Player_Item before any changes
    SELECT 'Before Insert' AS status, pi.player_item_id, p.player_name, i.item_name, pi.state
    FROM Player_Item pi
    JOIN Player p ON pi.player_id = p.player_id
    JOIN Item i ON pi.item_id = i.item_id
    WHERE pi.player_id = who_to_check;

    -- Attempt to insert a valid item
    create temporary table tmp as
    select min(item_id) as temp_value
    from player_item;

    CALL InsertPlayerItem(who_to_check, temp_value);

    -- Show the state of Player_Item after inserting a valid item
    SELECT 'After Inserting Valid Item' AS status, pi.player_item_id, p.player_name, i.item_name, pi.state
    FROM Player_Item pi
    JOIN Player p ON pi.player_id = p.player_id
    JOIN Item i ON pi.item_id = i.item_id
    WHERE pi.player_id = who_to_check;

    -- Attempt to insert an invalid item
    CALL InsertPlayerItem(who_to_check, temp_value-1);

    -- Show the state of Player_Item after attempting to insert the invalid item
    SELECT 'After Attempting to Insert Invalid Item' AS status, pi.player_item_id, p.player_name, i.item_name, pi.state
    FROM Player_Item pi
    JOIN Player p ON pi.player_id = p.player_id
    JOIN Item i ON pi.item_id = i.item_id
    WHERE pi.player_id = who_to_check;

    -- Attempt to change Player_Item state back to TRUE (which should not be possible due to the trigger)
    UPDATE Player_Item SET state = TRUE WHERE player_item_id = who_to_check;

    -- Check if Player_Item state remains FALSE
    SELECT 'After Attempting to Change Back' AS status, pi.player_item_id, p.player_name, i.item_name, pi.state
    FROM Player_Item pi
    JOIN Player p ON pi.player_id = p.player_id
    JOIN Item i ON pi.item_id = i.item_id
    WHERE pi.player_id = who_to_check;

    drop temporary table tmp;
END;
//
DELIMITER //

DROP TRIGGER IF EXISTS lock_weapon_state;
DELIMITER //
CREATE TRIGGER lock_weapon_state
BEFORE UPDATE ON Player_Item
FOR EACH ROW
BEGIN
    IF OLD.state = FALSE AND NEW.state = TRUE THEN
        SET NEW.state = FALSE; -- Prevent changing back to TRUE
    END IF;
END;
//
DELIMITER //

DROP PROCEDURE IF EXISTS Test_UnLockAchievementsIfInvalidItem;
DELIMITER //
CREATE PROCEDURE Test_UnLockAchievementsIfInvalidItem(in who_is_trying int)
BEGIN
    -- Show achievements before the update
    SELECT 'Before Update' AS status, ca.quest_id, ca.player_id, ca.requires_all_player_items
    FROM Check_Achievement ca
    JOIN Complete c ON ca.quest_id = c.quest_id
    WHERE c.player_id = who_is_trying;

    -- Update a valid player item to state FALSE
    UPDATE Player_Item SET state = FALSE WHERE player_item_id = who_is_trying;

    -- Show achievements after the update
    SELECT 'After Update' AS status, ca.quest_id, ca.player_id, ca.requires_all_player_items
    FROM Check_Achievement ca
    JOIN Complete c ON ca.quest_id = c.quest_id
    WHERE c.player_id = who_is_trying;

    -- Attempt to reset the state (this should fail due to the trigger)
    UPDATE Player_Item SET state = TRUE WHERE player_item_id = who_is_trying;

    -- Check the state remains FALSE
    SELECT 'After Attempting to Change Back' AS status, pi.player_item_id, p.player_name, i.item_name, pi.state
    FROM Player_Item pi
    JOIN Player p ON pi.player_id = p.player_id
    JOIN Item i ON pi.item_id = i.item_id
    WHERE pi.player_id = who_is_trying;
END;
//
DELIMITER //

-- Sample Data Insertion

-- Insert data into Item
INSERT INTO Item (item_name) VALUES
('Sword'), ('Shield'), ('Potion'), ('Helmet'), ('Armor'),
('Boots'), ('Ring'), ('Amulet'), ('Gloves'), ('Bow'),
('Dagger'), ('Staff'), ('Wand'), ('Axe'), ('Spear');

-- Insert data into Guild
INSERT INTO Guild (guild_name) VALUES
('Warriors'), ('Mages'), ('Hunters'), ('Thieves');

-- Insert data into Player
INSERT INTO Player (player_name, guild_id) VALUES
('Alice', 1), ('Bob', 2), ('Charlie', 3), ('Diana', 4),
('Eve', 1), ('Frank', 2), ('Grace', 3), ('Hank', 4),
('Ivy', 1), ('Jack', 2), ('Karen', 3), ('Leo', 4),
('Mona', 1), ('Nate', 2), ('Olivia', 3);

-- Insert data into NPC
INSERT INTO NPC (npc_name, guild_id) VALUES
('Gandalf', 1), ('Aragorn', 2), ('Legolas', 3), ('Gimli', 4),
('Frodo', 1), ('Sam', 2), ('Merry', 3), ('Pippin', 4),
('Boromir', NULL), ('Saruman', NULL), ('Elrond', NULL),
('Galadriel', NULL), ('Eowyn', NULL), ('Faramir', NULL),
('Theoden', NULL);

-- Insert data into Player_Item
INSERT INTO Player_Item (player_id, item_id, state) VALUES
(1, 1, TRUE), (1, 2, TRUE), (2, 3, TRUE), (2, 4, TRUE),
(3, 5, TRUE), (3, 6, TRUE), (4, 7, TRUE), (4, 8, TRUE),
(5, 9, TRUE), (5, 10, TRUE), (6, 11, TRUE), (6, 12, TRUE),
(7, 13, TRUE), (7, 14, TRUE), (8, 15, TRUE);

-- Insert data into Quest
INSERT INTO Quest (quest_name, state) VALUES
('Kill 10 Goblins',TRUE), ('Collect 5 Herbs',TRUE), ('Find the Lost Sword',TRUE),
('Rescue the Princess',TRUE), ('Defend the Castle',TRUE), ('Escort the Merchant',TRUE),
('Retrieve the Amulet',TRUE), ('Explore the Cave',TRUE), ('Defeat the Dragon',TRUE),
('Save the Village',TRUE), ('Hunt the Beast',TRUE), ('Recover the Artifact',TRUE),
('Protect the Caravan',TRUE), ('Locate the Hidden Treasure',TRUE), ('Destroy the Bandit Camp',TRUE);

-- Insert data into Dimension
INSERT INTO Dimension (dimension_name) VALUES
('Earth'), ('Mars'), ('Venus'), ('Jupiter');

-- Insert data into Achievement
INSERT INTO Achievement (achievement_name) VALUES
('First Blood'), ('Master Collector'), ('Quest Master'),
('Dimension Traveller'), ('Hero of the Realm'), ('Dragon Slayer'),
('Treasure Hunter'), ('Guild Champion'), ('Legendary Explorer'),
('Master of Magic'), ('Grand Alchemist'), ('Shadow Assassin'),
('Divine Protector'), ('Warrior of Light'), ('Master Blacksmith'),
('Noble Defender');

-- Insert data into Complete
INSERT INTO Complete (player_id, quest_id) VALUES
(1, 1), (1, 2), (2, 3), (2, 4), (3, 5), (3, 6), (4, 7), (4, 8),
(5, 9), (5, 10), (6, 11), (6, 12), (7, 13), (7, 14), (8, 15);

-- Insert data into Talks
INSERT INTO Talks (player_id, npc_id) VALUES
(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8),
(9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15);

-- Insert data into Initiate
INSERT INTO Initiate (npc_id, quest_id) VALUES
(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8),
(9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15);

-- Insert data into Check_Achievement
INSERT INTO Check_Achievement (quest_id, player_id, requires_all_player_items) VALUES
(1, 1, TRUE),
(2, 1, TRUE),
(3, 1, TRUE),
(4, 1, TRUE),
(5, 2, TRUE),
(6, 2, TRUE),
(7, 2, TRUE),
(8, 2, TRUE),
(9, 3, TRUE),
(10, 3, TRUE),
(11, 3, TRUE),
(12, 3, TRUE),
(13, 4, TRUE),
(14, 4, TRUE),
(15, 4, TRUE);

-- Insert data into Affiliated
INSERT INTO Affiliated (guild_id, npc_id, affiliation) VALUES
(1, 1, 'good'), (1, 5, 'good'), (1, 9, 'good'), (1, 13, 'good'),
(2, 2, 'neutral'), (2, 6, 'neutral'), (2, 10, 'neutral'), (2, 14, 'neutral'),
(3, 3, 'evil'), (3, 7, 'evil'), (3, 11, 'evil'), (3, 15, 'evil'),
(4, 4, 'neutral'), (4, 8, 'neutral'), (4, 12, 'neutral');

-- Insert data into Belongs
INSERT INTO Belongs (player_id, guild_id) VALUES
(1, 1), (2, 2), (3, 3), (4, 4), (5, 1), (6, 2), (7, 3), (8, 4),
(9, 1), (10, 2), (11, 3), (12, 4), (13, 1), (14, 2), (15, 3);

-- Insert data into Owns
INSERT INTO Owns (player_id, player_item_id) VALUES
(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8),
(9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15);

-- Insert data into Unlocks
INSERT INTO Unlocks (achievement_id, dimension_id) VALUES
(1, 1), (2, 1), (3, 1), (4, 1), (5, 2), (6, 2), (7, 2), (8, 2),
(9, 3), (10, 3), (11, 3), (12, 3), (13, 3), (14, 3), (15, 4);

-- Insert data into Travel
INSERT INTO Travel (player_id, dimension_id) VALUES
(1, 1), (2, 2), (3, 3), (4, 4), (5, 1), (6, 2), (7, 3), (8, 4),
(9, 1), (10, 2), (11, 3), (12, 4), (13, 1), (14, 2), (15, 3);

-- Test Procedures
#CALL ChangeDimension(3, 1);
#CALL Test_InsertPlayerItem(3);
#CALL Test_UnLockAchievementsIfInvalidItem(3);