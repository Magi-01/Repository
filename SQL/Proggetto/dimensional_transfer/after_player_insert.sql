create definer = root@localhost trigger after_player_insert
    after insert
    on player
    for each row
BEGIN
    INSERT INTO Completion (player_id, quest_id, state) VALUES (NEW.player_id, NEW.quest_id, FALSE);
END;

