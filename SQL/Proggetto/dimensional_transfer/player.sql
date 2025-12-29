create table player
(
    player_id int auto_increment
        primary key,
    name      varchar(255) not null,
    guild_id  int          null,
    quest_id  int          null,
    email     varchar(255) null,
    constraint player_ibfk_1
        foreign key (guild_id) references guild (guild_id),
    constraint player_ibfk_2
        foreign key (quest_id) references quest (quest_id)
);

create index idx_guild_id
    on player (guild_id);

create index idx_quest_id
    on player (quest_id);

