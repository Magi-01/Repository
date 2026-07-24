create definer = root@localhost view guildplayercount as
select `g`.`name` AS `guild_name`, count(`p`.`player_id`) AS `player_count`
from (`dimensional_transfer`.`guild` `g` left join `dimensional_transfer`.`player` `p`
      on ((`g`.`guild_id` = `p`.`guild_id`)))
group by `g`.`guild_id`;

