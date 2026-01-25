using System.Collections.Generic;

// Defines a simple, immutable data structure for a product
public record Product(string ProductName, decimal Price, int Quantity);

public static class Inventory
{
    // The C# representation of your data
    public static readonly List<Product> CurrentInventory = new List<Product>
    {
        new Product("Laptop", 1200.00m, 5),
        new Product("Mouse", 25.50m, 10),
        new Product("Keyboard", 75.00m, 8),
        new Product("Monitor", 300.00m, 3)
    };
}