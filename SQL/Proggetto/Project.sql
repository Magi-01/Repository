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

CALL ChangeDimension(3, 1);
CALL Test_InsertPlayerItem(3);
CALL Test_UnLockAchievementsIfInvalidItem(3);