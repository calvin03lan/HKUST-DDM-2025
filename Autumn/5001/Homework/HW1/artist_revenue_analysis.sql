-- 1. total revenue of every artist
WITH artist_revenue AS (
    SELECT
        ar.ArtistId,
        ar.Name AS Artist,
        SUM(il.UnitPrice * il.Quantity) AS Revenue
    FROM Artist ar
    JOIN Album al  ON al.ArtistId = ar.ArtistId
    JOIN Track tr  ON tr.AlbumId  = al.AlbumId
    JOIN InvoiceLine il ON il.TrackId = tr.TrackId
    GROUP BY ar.ArtistId, ar.Name
),

-- 2. Artist income percentile
artist_percentile AS (
    SELECT
        a1.ArtistId,
        ROUND(COUNT(CASE WHEN a2.Revenue <= a1.Revenue THEN 1 END) * 100.0 /
              (SELECT COUNT(*) FROM artist_revenue), 2) AS Percentile
    FROM artist_revenue a1
    CROSS JOIN artist_revenue a2
    GROUP BY a1.ArtistId
),

-- 3. count of albums of every artist
artist_album_count AS (
    SELECT ArtistId, COUNT(*) AS AlbumCount
    FROM Album
    GROUP BY ArtistId
),

-- 4. revenue of evevy album
album_revenue AS (
    SELECT
        al.AlbumId,
        al.ArtistId,
        SUM(il.UnitPrice * il.Quantity) AS AlbumRevenue
    FROM Album al
    JOIN Track tr  ON tr.AlbumId  = al.AlbumId
    JOIN InvoiceLine il ON il.TrackId = tr.TrackId
    GROUP BY al.AlbumId, al.ArtistId
),

-- 5. Proportion of an artist's best album revenue
artist_best_album AS (
    SELECT
        ArtistId,
        MAX(AlbumRevenue) AS BestAlbumRevenue
    FROM album_revenue
    GROUP BY ArtistId
),

-- 6. Number of artists that this artist outperforms
outperformed AS (
    SELECT
        a1.ArtistId,
        COUNT(a2.ArtistId) AS Outperformed
    FROM artist_revenue a1
    JOIN artist_revenue a2 ON a2.Revenue < a1.Revenue
    GROUP BY a1.ArtistId
),

-- 7. Summary
full_stats AS (
    SELECT
        ar.Artist,
        ap.Percentile,
        ROUND(ar.Revenue, 2) AS Revenue,
        COALESCE(aac.AlbumCount, 0) AS AlbumCount,
        ROUND(ar.Revenue / NULLIF(aac.AlbumCount, 0), 2) AS "Avg/Album",
        ROUND(aba.BestAlbumRevenue * 100.0 / NULLIF(ar.Revenue, 0), 2) AS BestPercent,
        COALESCE(o.Outperformed, 0) AS Outperformed
    FROM artist_revenue ar
    JOIN artist_percentile ap ON ap.ArtistId = ar.ArtistId
    LEFT JOIN artist_album_count aac ON aac.ArtistId = ar.ArtistId
    LEFT JOIN artist_best_album aba ON aba.ArtistId = ar.ArtistId
    LEFT JOIN outperformed o ON o.ArtistId = ar.ArtistId
)

-- 8. take the lead 20% and sort 
SELECT
    Artist,
    Percentile,
    Revenue,
    AlbumCount,
    "Avg/Album",
    BestPercent,
    Outperformed
FROM full_stats
WHERE Percentile >= 80
ORDER BY Revenue DESC;