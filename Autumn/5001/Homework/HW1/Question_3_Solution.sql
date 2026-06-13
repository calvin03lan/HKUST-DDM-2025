-- Calculate total revenue per artist
WITH artist_revenue AS (
    SELECT 
        ar.ArtistId,
        ar.Name AS ArtistName,
        ROUND(SUM(il.UnitPrice * il.Quantity), 2) AS TotalRevenue -- Round to 2 decimal places for nicer formatting
    FROM Artist ar
    JOIN Album al ON ar.ArtistId = al.ArtistId
    JOIN Track t ON al.AlbumId = t.AlbumId
    JOIN InvoiceLine il ON t.TrackId = il.TrackId
    GROUP BY ar.ArtistId, ar.Name
),
-- Add revenue rankings and percentiles
artist_rankings AS (
    SELECT 
        *,
        ROW_NUMBER() OVER (ORDER BY TotalRevenue DESC) AS RevenueRank,
        COUNT(*) OVER () AS TotalArtists, -- COUNT here is used as a window function, not an aggregate! Think about why this is needed.
        ROUND(
            100.0 * (COUNT(*) OVER () - ROW_NUMBER() OVER (ORDER BY TotalRevenue DESC) + 1) / COUNT(*) OVER (),
            1
        ) AS RevenuePercentile
    FROM artist_revenue
),
-- Calculate revenue per album for each artist
album_revenue AS (
    SELECT 
        ar.ArtistId,
        al.AlbumId,
        al.Title AS AlbumTitle,
        ROUND(SUM(il.UnitPrice * il.Quantity), 2) AS AlbumRevenue
    FROM Artist ar
    JOIN Album al ON ar.ArtistId = al.ArtistId
    JOIN Track t ON al.AlbumId = t.AlbumId
    JOIN InvoiceLine il ON t.TrackId = il.TrackId
    GROUP BY ar.ArtistId, al.AlbumId, al.Title
),
-- Calculate album statistics for each artist
artist_album_stats AS (
    SELECT 
        ArtistId,
        COUNT(*) AS NumberOfAlbums,
        ROUND(AVG(AlbumRevenue), 2) AS AvgRevenuePerAlbum,
        MAX(AlbumRevenue) AS BestAlbumRevenue
    FROM album_revenue
    GROUP BY ArtistId
),
-- Filter to include only top 20% of artists
top_20_percent AS (
    SELECT *
    FROM artist_rankings
    WHERE RevenuePercentile >= 80.0
)

-- Final, main query
SELECT 
    t20.ArtistName AS "Artist",
    t20.RevenuePercentile AS "Percentile",
    t20.TotalRevenue AS "Revenue",
    aas.NumberOfAlbums AS "AlbumCount",
    aas.AvgRevenuePerAlbum AS "Avg/Album",
    ROUND(
        (aas.BestAlbumRevenue * 100.0) / t20.TotalRevenue, 
        1
    ) AS "BestPercent",
    (t20.TotalArtists - t20.RevenueRank) AS "Outperformed"
FROM top_20_percent t20
JOIN artist_album_stats aas ON t20.ArtistId = aas.ArtistId
ORDER BY t20.TotalRevenue DESC;