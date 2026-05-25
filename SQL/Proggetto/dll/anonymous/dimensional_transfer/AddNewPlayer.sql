create
    definer = root@localhost procedure AddNewPlayer(IN p_name varchar(255), IN p_guild_id int, IN p_quest_id int)
BEGIN
    INSERT INTO Player (name, guild_id, quest_id) VALUES (p_name, p_guild_id, p_quest_id);
END;

