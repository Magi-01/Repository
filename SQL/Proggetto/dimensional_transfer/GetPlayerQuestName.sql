create
    definer = root@localhost function GetPlayerQuestName(player_id int) returns varchar(255) deterministic
BEGIN
    DECLARE quest_name VARCHAR(255);
    SELECT q.name INTO quest_name
    FROM Player p
    JOIN Quest q ON p.quest_id = q.quest_id
    WHERE p.player_id = player_id;
    RETURN quest_name;
END;

