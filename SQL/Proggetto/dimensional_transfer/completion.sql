create table completion
(
    player_id int                  not null,
    quest_id  int                  not null,
    state     tinyint(1) default 0 null,
    primary key (player_id, quest_id),
    constraint completion_ibfk_1
        foreign key (player_id) references player (player_id)
            on update cascade on delete cascade,
    constraint completion_ibfk_2
        foreign key (quest_id) references quest (quest_id)
            on update cascade on delete cascade
);

create index quest_id
    on completion (quest_id);

