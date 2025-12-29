create
    definer = root@localhost function GetGuildName(guild_id int) returns varchar(255) deterministic
BEGIN
    DECLARE guild_name VARCHAR(255);
    SELECT name INTO guild_name FROM Guild WHERE guild_id = guild_id;
    RETURN guild_name;
END;

