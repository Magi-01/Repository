use classicmodels;

select * from classicmodels.products
where length(productName) >= 15;

select * from orders
where month(orderDate) = 01;

select * from products
order by MSRP;

select country,creditLimit
from customers
order by country, creditLimit DESC;

select *
from orders
order by field(status,'shipped','Cancelled','On Hold','Resolved','Disputed','In Process');

select productName,
MSRP-buyPrice as margine
from products
where MSRP < 100
order by margine DESC;

select country
from customers
order by country
