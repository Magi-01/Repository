create function multiply (val int)
returns int
not deterministic
begin
    return 2*val;
end;


create function credit_level (val int)
returns varchar(245)
deterministic
begin
    if val < 25 then return 'bronze';
    end if;
    if val >= 25 and val < 50 then return 'silver';
    end if;
    if val >= 50 and val < 75 then return 'gold';
    end if;
    if val >= 75 then return 'platinum';
    end if;
end

