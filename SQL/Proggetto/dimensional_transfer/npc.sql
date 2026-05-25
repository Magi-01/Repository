create table npc
(
    npc_id   int auto_increment
        primary key,
    guild_id int          null,
    name     varchar(255) not null,
    constraint npc_ibfk_1
        foreign key (guild_id) references guild (guild_id)
);

create index guild_id
    on npc (guild_id);

