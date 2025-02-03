                       -- Project: AI-Powered Dynamic Renewable Energy Grid for Smart Cities--
                       
                       
-- WORKING-- 

-- Query to retrieve daily total renewable energy generated
SELECT Date, SUM(EnergyGenerated_kWh) AS TotalEnergy 
FROM renewable_energy 
GROUP BY Date 
ORDER BY Date;

-- Query to find peak grid load time
SELECT Date, Time, GridLoad_MW 
FROM grid_load 
ORDER BY GridLoad_MW DESC 
LIMIT 1;

-- Query to analyze energy consumption trends
SELECT ConsumerType, AVG(EnergyConsumed_kWh) AS AvgConsumption 
FROM energy_consumption 
GROUP BY ConsumerType;



-- This query will show the Net Energy produced vs consumed for each energy source--
SELECT 
    r.EnergySource, 
    SUM(r.EnergyGenerated_kWh) AS TotalEnergyProduced, 
    COALESCE(SUM(e.EnergyConsumed_kWh), 0) AS TotalEnergyConsumed, 
    (SUM(r.EnergyGenerated_kWh) - COALESCE(SUM(e.EnergyConsumed_kWh), 0)) AS NetEnergy 
FROM renewable_energy r
LEFT JOIN energy_consumption e 
    ON r.Date = e.Date AND r.Time = e.Time
GROUP BY r.EnergySource
ORDER BY NetEnergy DESC;

-- Query to Get Total Energy Consumed for Each Consumer Type--

SELECT 
    e.ConsumerType, 
    SUM(e.EnergyConsumed_kWh) AS TotalEnergyConsumed
FROM energy_consumption e
GROUP BY e.ConsumerType
ORDER BY TotalEnergyConsumed DESC;



-- Query to Calculate Grid Load Efficiency:--

SELECT Date , TIME, GridLoad_MW, BatteryStorage_MWh, 
    (GridLoad_MW / BatteryStorage_MWh) AS LoadEfficiency
FROM grid_load;



---------- --------------------- --------------------------- - ----------------------- -----  ------------------------           ----------------



























































