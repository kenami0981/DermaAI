using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace Derma.Infrastructure.Migrations
{
    /// <inheritdoc />
    public partial class AddAnalysisResultAndDetections : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropColumn(
                name: "acneLevel",
                table: "SkinAnalysis");

            migrationBuilder.DropColumn(
                name: "confidence",
                table: "SkinAnalysis");

            migrationBuilder.AddColumn<int>(
                name: "Status",
                table: "SkinAnalysis",
                type: "INTEGER",
                nullable: false,
                defaultValue: 0);
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropColumn(
                name: "Status",
                table: "SkinAnalysis");

            migrationBuilder.AddColumn<string>(
                name: "acneLevel",
                table: "SkinAnalysis",
                type: "TEXT",
                nullable: true);

            migrationBuilder.AddColumn<double>(
                name: "confidence",
                table: "SkinAnalysis",
                type: "REAL",
                nullable: true);
        }
    }
}
