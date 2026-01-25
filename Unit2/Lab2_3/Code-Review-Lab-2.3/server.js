// Initialize database once to prevent connection leaks
const db = new sqlite3.Database('users.db');

app.get('/user', (req, res) => {
    const userId = req.query.id;

    // Validate the input
    if (!userId || isNaN(parseInt(userId))) {
        return res.status(400).send('Invalid or missing User ID');
    }

    // Use parameterized query to prevent SQL injection
    const query = "SELECT * FROM users WHERE user_id = ?";

    db.get(query, [userId], (err, row) => {
        if (err) {
            console.error(err); // Log internally
            return res.status(500).send('Internal Server Error'); // Generic message for security
        }

        if (!row) {
            return res.status(404).send('User not found');
        }
        
        // Use descriptive variable names
        const userProfile = processData(row);
        const jsonResponse = formatResponse(userProfile);
        
        res.send(jsonResponse);
    });
});
