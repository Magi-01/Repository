create
    definer = root@localhost procedure ListPlayers()
BEGIN
    DECLARE done INT DEFAULT 0;
    DECLARE p_name VARCHAR(255);
    DECLARE cur CURSOR FOR SELECT name FROM Player;
    DECLARE CONTINUE HANDLER FOR NOT FOUND SET done = 1;

    OPEN cur;

    read_loop: LOOP
        FETCH cur INTO p_name;
        IF done THEN
            LEAVE read_loop;
        END IF;
        SELECT p_name;
    END LOOP;

    CLOSE cur;
END;

