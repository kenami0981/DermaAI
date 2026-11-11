using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace Derma.Infrastructure.Migrations
{
    /// <inheritdoc />
    public partial class SyncCheck : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropColumn(
                name: "Value",
                table: "SkinAnalysisTag");
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.AddColumn<bool>(
                name: "Value",
                table: "SkinAnalysisTag",
                type: "INTEGER",
                nullable: false,
                defaultValue: false);
        }
    }
}
