create definer = root@localhost view playerquestinfo as
select `p`.`name` AS `player_name`, `q`.`name` AS `quest_name`, `q`.`state` AS `quest_state`
from (`dimensional_transfer`.`player` `p` join `dimensional_transfer`.`quest` `q`
      on ((`p`.`quest_id` = `q`.`quest_id`)));

