[Authorize]
[HttpGet("/user")]
public async Task<IActionResult> GetUserProfileAsync()
{
    // Validate and parse the user ID parameter
    if (!Request.Query.TryGetValue("id", out var userIdParam) || string.IsNullOrWhiteSpace(userIdParam))
    {
        return BadRequest(new { error = "Missing or invalid 'id' parameter." });
    }

    if (!int.TryParse(userIdParam, out int userId))
    {
        return BadRequest(new { error = "Invalid 'id' parameter. Must be an integer." });
    }

    try
    {
        // Get connection string from configuration
        var connectionString = _configuration.GetConnectionString("DefaultConnection");
        
        using (var connection = new SqliteConnection(connectionString))
        {
            await connection.OpenAsync();
            var command = connection.CreateCommand();
            
            // Use parameterized query to prevent SQL injection
            command.CommandText = "SELECT * FROM users WHERE user_id = @userId";
            command.Parameters.AddWithValue("@userId", userId);
            
            using (var reader = await command.ExecuteReaderAsync())
            {
                if (await reader.ReadAsync())
                {
                    var userData = ProcessData(reader);
                    var profileResponse = FormatResponse(userData);
                    return Ok(profileResponse);
                }
            }
        }
        return NotFound(new { error = "User not found." });
    }
    catch (SqliteException ex)
    {
        _logger.LogError(ex, "Database error while retrieving user profile.");
        return StatusCode(500, new { error = "Database error occurred." });
    }
    catch (Exception ex)
    {
        _logger.LogError(ex, "Unexpected error while retrieving user profile.");
        return StatusCode(500, new { error = "An unexpected error occurred." });
    }
}
