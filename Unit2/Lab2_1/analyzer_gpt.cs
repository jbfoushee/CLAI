using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Globalization;

/// <summary>
/// Data model representing a single sales item with product name, price, and quantity.
/// </summary>
public record SalesItem(string ProductName, decimal Price, int Quantity);

/// <summary>
/// SalesAnalyzer class provides methods to load sales data, calculate total sales, and find the top-selling product.
/// </summary>
public class SalesAnalyzer
{
    private string _filename;
    private List<SalesItem> _salesData;

    /// <summary>
    /// Initializes a new instance of the SalesAnalyzer class with the specified CSV filename.
    /// </summary>
    /// <param name="filename">The path to the CSV file containing sales data.</param>
    public SalesAnalyzer(string filename)
    {
        _filename = filename;
        _salesData = new List<SalesItem>();
    }

    /// <summary>
    /// Loads sales data from the CSV file into memory.
    /// Expected CSV format: product_name,price,quantity (with header row).
    /// Skips malformed rows and logs errors for diagnostic purposes.
    /// </summary>
    /// <returns>True if data was loaded successfully, false otherwise.</returns>
    public bool LoadData()
    {
        _salesData.Clear();
        try
        {
            var dataLines = File.ReadLines(_filename).Skip(1);
            foreach (string line in dataLines)
            {
                string[] columns = line.Split(',');
                if (columns.Length == 3)
                {
                    if (decimal.TryParse(columns[1], NumberStyles.Any, CultureInfo.InvariantCulture, out decimal price) &&
                        int.TryParse(columns[2], out int quantity))
                    {
                        _salesData.Add(new SalesItem(columns[0].Trim(), price, quantity));
                    }
                    else
                    {
                        Console.WriteLine($"Skipping malformed data in row: {line}");
                    }
                }
            }
            return true;
        }
        catch (FileNotFoundException)
        {
            Console.WriteLine($"Error: The file '{_filename}' was not found.");
            return false;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"An unexpected error occurred while loading data: {ex.Message}");
            return false;
        }
    }

    /// <summary>
    /// Calculates the total sales revenue from all loaded sales items.
    /// Revenue is computed as price * quantity for each product.
    /// </summary>
    /// <returns>The total sales revenue as a decimal value. Returns 0 if no data is loaded.</returns>
    public decimal CalculateTotalSales()
    {
        if (_salesData.Count == 0)
        {
            Console.WriteLine("No sales data available. Please load data first.");
            return 0.0m;
        }
        return _salesData.Sum(item => item.Price * item.Quantity);
    }

    /// <summary>
    /// Identifies the top-selling product by total revenue (price * quantity).
    /// </summary>
    /// <returns>The SalesItem with the highest total revenue, or null if no data is loaded.</returns>
    public SalesItem? FindTopProduct()
    {
        if (_salesData.Count == 0)
        {
            Console.WriteLine("No sales data available. Please load data first.");
            return null;
        }
        return _salesData.OrderByDescending(item => item.Price * item.Quantity).FirstOrDefault();
    }

    /// <summary>
    /// Main entry point. Creates a SalesAnalyzer instance, loads data, and displays analysis results.
    /// </summary>
    public static void Main(string[] args)
    {
        string salesDataFile = "sales_data.csv";

        // Create a sample file if it doesn't exist for testing purposes
        if (!File.Exists(salesDataFile))
        {
            File.WriteAllText(salesDataFile, "product_name,price,quantity\nLaptop,1200.00,5\nMouse,25.50,10\n");
        }

        // Instantiate the analyzer and load data
        var analyzer = new SalesAnalyzer(salesDataFile);
        if (!analyzer.LoadData())
        {
            return;
        }

        // Calculate and display total sales
        decimal totalSales = analyzer.CalculateTotalSales();
        Console.WriteLine($"=== Sales Analysis Report ===\n");
        Console.WriteLine($"Total sales from {salesDataFile}: {totalSales.ToString("C", CultureInfo.CurrentCulture)}");

        // Find and display top-selling product
        var topProduct = analyzer.FindTopProduct();
        if (topProduct != null)
        {
            decimal revenue = topProduct.Price * topProduct.Quantity;
            Console.WriteLine($"\nTop-selling product by revenue: {topProduct.ProductName}");
            Console.WriteLine($"  Price: {topProduct.Price.ToString("C", CultureInfo.CurrentCulture)}");
            Console.WriteLine($"  Quantity Sold: {topProduct.Quantity}");
            Console.WriteLine($"  Total Revenue: {revenue.ToString("C", CultureInfo.CurrentCulture)}");
        }
        else
        {
            Console.WriteLine("No sales data available to determine top-selling product.");
        }
    }
}