create table earn
(
    player_id      int                  not null,
    achievement_id int                  not null,
    state          tinyint(1) default 0 null,
    primary key (player_id, achievement_id),
    constraint earn_ibfk_1
        foreign key (player_id) references player (player_id)
            on update cascade on delete cascade,
    constraint earn_ibfk_2
        foreign key (achievement_id) references achievement (achievement_id)
            on update cascade on delete cascade
);

create index achievement_id
    on earn (achievement_id);

