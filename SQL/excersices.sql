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

select distinct city
from customers
order by city;

select customerNumber, salesRepEmployeeNumber, firstName, lastName, employeeNumber
from employees
inner join customers
On salesrepemployeenumber = employeenumber;

select productCode, textDescription, productName
from products inner join productlines
On products.productLine = productlines.productLine;

select employeeNumber,firstName,lastName,officeCode,city
from employees inner join offices
using(officeCode);

select customerNumber, customerName, salesRepEmployeeNumber, firstName, lastName
from customers left outer join employees
on employeeNumber = salesRepEmployeeNumber

